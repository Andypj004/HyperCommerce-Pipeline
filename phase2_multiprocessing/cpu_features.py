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
import logging
import warnings
import glob
from math import log2
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from multiprocessing import Pool, cpu_count, current_process

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FASE-2] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Función de trabajo (se ejecuta en cada proceso hijo) ─────────────────────

def _compute_user_features(chunk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula ~10 features por usuario para todos los usuarios del chunk.
    Esta función corre en un proceso hijo independiente.
    """
    proc = current_process().name

    records = []
    for user_id, grp in chunk_df.groupby("user_id", sort=False):
        n = len(grp)

        # ── Monetary ──────────────────────────────────────────────────────
        total_spend   = grp["price"].sum()
        avg_price     = grp["price"].mean()
        std_price     = grp["price"].std() if n > 1 else 0.0
        min_price     = grp["price"].min()
        max_price     = grp["price"].max()

        # ── Frequency ─────────────────────────────────────────────────────
        frequency     = n

        # ── Recency proxy (hora más reciente del dataset como referencia) ──
        # (no tenemos fecha de referencia externa; usamos month como proxy)
        last_month    = int(grp["month"].max())
        first_month   = int(grp["month"].min())
        span_months   = max(last_month - first_month, 1)

        # ── Diversidad de categorías (entropía de Shannon) ─────────────────
        cat_counts    = grp["category_top"].value_counts(normalize=True)
        entropy_cat   = -sum(p * log2(p) for p in cat_counts if p > 0)

        # ── Marca favorita y lealtad ───────────────────────────────────────
        brand_counts  = grp["brand"].value_counts()
        top_brand     = brand_counts.index[0] if len(brand_counts) > 0 else "unknown"
        brand_loyalty = brand_counts.iloc[0] / n  # % compras a marca top

        # ── Hora de compra preferida ───────────────────────────────────────
        peak_hour     = int(grp["hour"].mode().iloc[0]) if n > 0 else 0

        # ── Ratio fin de semana (dow 5=sab, 6=dom) ─────────────────────────
        weekend_ratio = (grp["dow"] >= 5).mean()

        # ── Bucket de precio predominante ─────────────────────────────────
        dominant_bucket = (
            grp["price_bucket"].value_counts().index[0]
            if "price_bucket" in grp.columns else "unknown"
        )

        records.append({
            "user_id":        user_id,
            "frequency":      frequency,
            "total_spend":    round(float(total_spend), 4),
            "avg_price":      round(float(avg_price), 4),
            "std_price":      round(float(std_price), 4),
            "min_price":      round(float(min_price), 4),
            "max_price":      round(float(max_price), 4),
            "entropy_cat":    round(float(entropy_cat), 6),
            "top_brand":      top_brand,
            "brand_loyalty":  round(float(brand_loyalty), 4),
            "peak_hour":      peak_hour,
            "weekend_ratio":  round(float(weekend_ratio), 4),
            "span_months":    span_months,
            "last_month":     last_month,
            "dominant_bucket":dominant_bucket,
        })

    result = pd.DataFrame(records)
    log.debug("%s — procesó %d usuarios", proc, len(result))
    return result


def _load_parquet_as_chunks() -> list[pd.DataFrame]:
    """
    Lee el Parquet de la Fase 1 en chunks de CHUNK_SIZE filas.
    Usa pyarrow directamente para proyección de columnas (solo lee lo necesario).
    """
    NEEDED_COLS = [
        "user_id", "product_id", "price",
        "category_top", "brand", "price_bucket",
        "hour", "dow", "month",
    ]

    parquet_files = sorted(
        glob.glob(os.path.join(config.PARQUET_DIR, "**", "*.parquet"), recursive=True)
    )
    if not parquet_files:
        raise FileNotFoundError(f"No se encontraron archivos Parquet en {config.PARQUET_DIR}")

    log.info("Particiones Parquet: %d", len(parquet_files))

    # Leer todo en un solo DataFrame (el Parquet ya está comprimido y filtrado)
    # pero en chunks para no saturar RAM
    chunks = []
    batch_rows = config.CHUNK_SIZE * config.CPU_WORKERS  # Lee N*W filas a la vez

    for fpath in parquet_files:
        table = pq.read_table(fpath, columns=NEEDED_COLS)
        df    = table.to_pandas()

        # Dividir en sub-chunks del tamaño correcto
        for start in range(0, len(df), batch_rows):
            chunks.append(df.iloc[start:start + batch_rows].copy())

        del table, df  # Liberar inmediatamente

    log.info("Total de chunks a procesar: %d", len(chunks))
    return chunks


def _merge_chunk_results(partial_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatena resultados parciales y resuelve duplicados de user_id
    (un usuario puede aparecer en varios chunks/particiones).
    """
    combined = pd.concat(partial_dfs, ignore_index=True)

    # Agrupar de nuevo para fusionar usuarios duplicados entre chunks
    agg = combined.groupby("user_id", as_index=False).agg(
        frequency      = ("frequency",     "sum"),
        total_spend    = ("total_spend",   "sum"),
        avg_price      = ("avg_price",     "mean"),
        std_price      = ("std_price",     "mean"),
        min_price      = ("min_price",     "min"),
        max_price      = ("max_price",     "max"),
        entropy_cat    = ("entropy_cat",   "mean"),
        top_brand      = ("top_brand",     lambda x: x.mode().iloc[0]),
        brand_loyalty  = ("brand_loyalty", "mean"),
        peak_hour      = ("peak_hour",     lambda x: x.mode().iloc[0]),
        weekend_ratio  = ("weekend_ratio", "mean"),
        span_months    = ("span_months",   "max"),
        last_month     = ("last_month",    "max"),
        dominant_bucket= ("dominant_bucket",lambda x: x.mode().iloc[0]),
    )
    return agg


# ── Punto de entrada ─────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    """
    Ejecuta la Fase 2 completa y devuelve el DataFrame de features por usuario.
    """
    t0 = time.perf_counter()
    log.info("Iniciando Fase 2 — %d workers CPU", config.CPU_WORKERS)

    chunks = _load_parquet_as_chunks()

    # Pool de procesos — cada proceso hijo recibe un chunk completo
    partial_results = []
    with Pool(processes=config.CPU_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(_compute_user_features, chunks)):
            partial_results.append(result)
            if (i + 1) % 5 == 0 or (i + 1) == len(chunks):
                log.info("  Chunks completados: %d / %d", i + 1, len(chunks))

    log.info("Fusionando resultados de %d chunks …", len(partial_results))
    features_df = _merge_chunk_results(partial_results)

    log.info("Features extraídas para %d usuarios únicos", len(features_df))

    # Guardar en Parquet para Fase 3
    features_df.to_parquet(config.CPU_RESULT, index=False, compression="snappy")
    log.info("Features guardadas en %s", config.CPU_RESULT)

    elapsed = time.perf_counter() - t0
    log.info("FASE 2 completada en %.1f s", elapsed)
    features_df.attrs["phase2_elapsed_s"] = round(elapsed, 2)
    return features_df


if __name__ == "__main__":
    df = run()
    print(df.describe())
