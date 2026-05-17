"""
phase2_multiprocessing/cpu_features.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FASE 2 — Extracción de Características Paralela en CPU (multiprocessing)
─────────────────────────────────────────────────────────────────────────
¿Qué procesamiento es adecuado para este dataset?

  Este dataset es de comportamiento de e-commerce (eventos de compra).
  El análisis más valioso es POR USUARIO: entender el perfil de gasto,
  la diversidad de categorías, la frecuencia de compra y la lealtad de marca.
  Estas métricas son independientes entre usuarios → paralelización perfecta.

  Cada proceso hijo recibe un chunk de user_ids y calcula:
    • RFM simplificado (Recency proxy, Frequency, Monetary)
    • Diversidad de categorías (entropía de Shannon)
    • Precio promedio, std, min, max por usuario
    • Marca favorita y lealtad de marca (% compras a marca top)
    • Hora de compra favorita (peak_hour)
    • Ratio compras fin de semana vs semana

  El resultado: una tabla de features por usuario, lista para:
    → Fase 3: construir la matriz de similitud entre usuarios en GPU
    → Fase 4: segmentación y visualización

Paralelismo:
  • multiprocessing.Pool distribuye chunks de ~50 000 filas entre
    todos los núcleos disponibles menos uno.
  • Cada proceso es independiente (no comparte memoria) → sin GIL.
  • Los resultados parciales se reciben por cola y se concatenan al final.
"""
import os
import sys
import time
import glob
import logging
import warnings
from multiprocessing import Pool, current_process

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FASE-2] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Columnas que cada worker necesita leer
NEEDED_COLS = [
    "user_id", "product_id", "price",
    "category_top", "brand", "price_bucket",
    "hour", "dow", "month",
]


# ── Worker: procesa UN archivo Parquet completo ───────────────────────────────

def _process_parquet_file(parquet_path: str) -> pd.DataFrame:
    """
    Lee un archivo Parquet y calcula todas las features por usuario
    usando EXCLUSIVAMENTE operaciones vectorizadas de pandas/numpy.

    Sin loops Python → el cómputo ocurre en C/Cython internamente.
    """
    proc = current_process().name

    # Leer solo columnas necesarias (proyección PyArrow → menos I/O y RAM)
    table = pq.read_table(parquet_path, columns=NEEDED_COLS)
    df    = table.to_pandas()
    del table  # liberar memoria PyArrow inmediatamente

    if df.empty:
        return pd.DataFrame()

    # Asegurar tipos correctos
    df["price"]    = df["price"].astype(np.float32)
    df["user_id"]  = df["user_id"].astype(np.int32)
    df["hour"]     = df["hour"].astype(np.int8)
    df["dow"]      = df["dow"].astype(np.int8)
    df["month"]    = df["month"].astype(np.int8)

    g = df.groupby("user_id", sort=False)

    # ── 1. Métricas monetarias y de frecuencia (una sola pasada groupby) ───
    price_agg = g["price"].agg(
        frequency  = "count",
        total_spend= "sum",
        avg_price  = "mean",
        std_price  = "std",
        min_price  = "min",
        max_price  = "max",
    ).reset_index()
    price_agg["std_price"] = price_agg["std_price"].fillna(0.0)

    # ── 2. Recency proxy (span de meses) ────────────────────────────────────
    month_agg = g["month"].agg(
        last_month = "max",
        first_month= "min",
    ).reset_index()
    month_agg["span_months"] = (
        month_agg["last_month"] - month_agg["first_month"]
    ).clip(lower=1)

    # ── 3. Entropía de Shannon de categorías (vectorizado) ──────────────────
    # Conteo de cada (user_id, category_top)
    cat_counts = (
        df.groupby(["user_id", "category_top"], sort=False)
        .size()
        .reset_index(name="cnt")
    )
    # Normalizar dentro de cada usuario
    user_totals = cat_counts.groupby("user_id")["cnt"].transform("sum")
    cat_counts["p"] = cat_counts["cnt"] / user_totals
    # Entropía: -Σ p·log2(p)
    cat_counts["h"] = -cat_counts["p"] * np.log2(cat_counts["p"].clip(lower=1e-10))
    entropy_agg = (
        cat_counts.groupby("user_id")["h"]
        .sum()
        .reset_index()
        .rename(columns={"h": "entropy_cat"})
    )

    # ── 4. Marca favorita y lealtad (vectorizado) ───────────────────────────
    brand_counts = (
        df.groupby(["user_id", "brand"], sort=False)
        .size()
        .reset_index(name="brand_cnt")
    )
    # Fila con máximo conteo por usuario (= marca favorita)
    top_brand_idx = brand_counts.groupby("user_id")["brand_cnt"].idxmax()
    top_brand_df  = brand_counts.loc[top_brand_idx, ["user_id", "brand", "brand_cnt"]].copy()
    top_brand_df  = top_brand_df.rename(columns={"brand": "top_brand",
                                                   "brand_cnt": "top_brand_cnt"})
    # Lealtad = compras a marca top / total compras
    top_brand_df  = top_brand_df.merge(
        price_agg[["user_id", "frequency"]], on="user_id", how="left"
    )
    top_brand_df["brand_loyalty"] = (
        top_brand_df["top_brand_cnt"] / top_brand_df["frequency"]
    ).clip(0, 1)
    top_brand_df = top_brand_df[["user_id", "top_brand", "brand_loyalty"]]

    # ── 5. Hora pico (moda vectorizada con value_counts) ────────────────────
    hour_counts = (
        df.groupby(["user_id", "hour"], sort=False)
        .size()
        .reset_index(name="h_cnt")
    )
    peak_hour_df = (
        hour_counts.loc[hour_counts.groupby("user_id")["h_cnt"].idxmax(),
                        ["user_id", "hour"]]
        .rename(columns={"hour": "peak_hour"})
    )

    # ── 6. Ratio fin de semana ───────────────────────────────────────────────
    df["is_weekend"] = (df["dow"] >= 5).astype(np.float32)
    weekend_agg = (
        g["is_weekend"]
        .mean()
        .reset_index()
        .rename(columns={"is_weekend": "weekend_ratio"})
    )

    # ── 7. Bucket de precio dominante ───────────────────────────────────────
    bucket_counts = (
        df.groupby(["user_id", "price_bucket"], sort=False)
        .size()
        .reset_index(name="b_cnt")
    )
    dominant_bucket_df = (
        bucket_counts.loc[bucket_counts.groupby("user_id")["b_cnt"].idxmax(),
                          ["user_id", "price_bucket"]]
        .rename(columns={"price_bucket": "dominant_bucket"})
    )

    # ── Merge de todas las features ──────────────────────────────────────────
    result = (
        price_agg
        .merge(month_agg[["user_id", "last_month", "span_months"]], on="user_id", how="left")
        .merge(entropy_agg,       on="user_id", how="left")
        .merge(top_brand_df,      on="user_id", how="left")
        .merge(peak_hour_df,      on="user_id", how="left")
        .merge(weekend_agg,       on="user_id", how="left")
        .merge(dominant_bucket_df,on="user_id", how="left")
    )

    # Redondear floats
    float_cols = ["total_spend","avg_price","std_price","min_price",
                  "max_price","entropy_cat","brand_loyalty","weekend_ratio"]
    result[float_cols] = result[float_cols].round(4)

    log.info("%s — %s: %d usuarios procesados",
             proc, os.path.basename(parquet_path), len(result))
    return result


# ── Merge final: fusionar resultados de múltiples archivos ───────────────────

def _merge_results(partial_dfs: list) -> pd.DataFrame:
    """
    Concatena los DataFrames de todos los workers y resuelve duplicados
    de user_id que aparecen en más de un archivo Parquet.
    Usa operaciones pandas vectorizadas (sin lambdas por fila).
    """
    combined = pd.concat([d for d in partial_dfs if not d.empty],
                         ignore_index=True)

    if combined.empty:
        return combined

    # Usuarios que aparecen en un solo archivo: no necesitan merge
    user_counts = combined["user_id"].value_counts()
    single_mask = combined["user_id"].isin(user_counts[user_counts == 1].index)
    single_df   = combined[single_mask].copy()
    multi_df    = combined[~single_mask].copy()

    if multi_df.empty:
        return single_df.reset_index(drop=True)

    # Para usuarios duplicados: agg vectorizado
    g = multi_df.groupby("user_id", sort=False)

    numeric_agg = g.agg(
        frequency      = ("frequency",     "sum"),
        total_spend    = ("total_spend",   "sum"),
        avg_price      = ("avg_price",     "mean"),
        std_price      = ("std_price",     "mean"),
        min_price      = ("min_price",     "min"),
        max_price      = ("max_price",     "max"),
        entropy_cat    = ("entropy_cat",   "mean"),
        brand_loyalty  = ("brand_loyalty", "mean"),
        weekend_ratio  = ("weekend_ratio", "mean"),
        span_months    = ("span_months",   "max"),
        last_month     = ("last_month",    "max"),
        peak_hour      = ("peak_hour",     "first"),   # aproximación rápida
    ).reset_index()

    # top_brand y dominant_bucket: el del archivo con más frecuencia
    top_brand_df = (
        multi_df.loc[multi_df.groupby("user_id")["frequency"].idxmax(),
                     ["user_id", "top_brand", "dominant_bucket"]]
    )
    numeric_agg = numeric_agg.merge(top_brand_df, on="user_id", how="left")

    merged = pd.concat([single_df, numeric_agg], ignore_index=True)
    return merged.reset_index(drop=True)


# ── Punto de entrada ──────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    t0 = time.perf_counter()

    # Encontrar todos los archivos Parquet de la Fase 1
    parquet_files = sorted(
        glob.glob(os.path.join(config.PARQUET_DIR, "**", "*.parquet"),
                  recursive=True)
    )
    if not parquet_files:
        raise FileNotFoundError(
            f"No se encontraron Parquets en {config.PARQUET_DIR}. "
            "Ejecuta primero la Fase 1."
        )

    log.info("Archivos Parquet: %d | Workers CPU: %d",
             len(parquet_files), config.CPU_WORKERS)

    # Distribuir archivos entre workers (uno por proceso)
    # imap_unordered → el proceso principal recibe resultados
    # en cuanto cada worker termina, sin esperar al más lento
    partial_results = []
    with Pool(processes=config.CPU_WORKERS) as pool:
        for i, result in enumerate(
            pool.imap_unordered(_process_parquet_file, parquet_files)
        ):
            partial_results.append(result)
            log.info("  Archivos completados: %d / %d", i + 1, len(parquet_files))

    log.info("Fusionando resultados …")
    features_df = _merge_results(partial_results)

    log.info("Features extraídas: %d usuarios únicos", len(features_df))
    log.info("  Columnas: %s", list(features_df.columns))

    features_df.to_parquet(config.CPU_RESULT, index=False, compression="snappy")
    log.info("Guardado en %s", config.CPU_RESULT)

    elapsed = time.perf_counter() - t0
    log.info("FASE 2 completada en %.1f s  (%.1f min)", elapsed, elapsed / 60)
    features_df.attrs["phase2_elapsed_s"] = round(elapsed, 2)
    return features_df


if __name__ == "__main__":
    df = run()
    print(df.describe())