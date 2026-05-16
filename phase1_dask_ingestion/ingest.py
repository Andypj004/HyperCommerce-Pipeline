"""
phase1_dask_ingestion/ingest.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FASE 1 — Ingesta y Preprocesamiento Out-of-Core con Dask
─────────────────────────────────────────────────────────
Estrategia de bajo consumo de RAM (para máquinas con ≤ 8 GB RAM):
  • Scheduler "synchronous" → sin workers distribuidos, sin duplicar
    datos entre procesos. Un hilo, RAM mínima.
  • Un CSV a la vez, blocksize="128MB" → máximo ~300 MB activos en RAM.
  • Cada partición se materializa, se transforma y se descarta.
  • Aggregations INCREMENTALES con acumuladores Python puros → nunca
    se retiene todo el dataset en memoria.
  • Parquet particionado por archivo fuente con PyArrow streaming writer.

RAM máxima estimada: ~600–800 MB.
"""

import os
import sys
import glob
import json
import logging
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FASE-1] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DTYPE_MAP = {
    "event_type":    "category",
    "product_id":    "int32",
    "category_id":   "int64",
    "category_code": "category",
    "brand":         "category",
    "price":         "float32",
    "user_id":       "int32",
    "user_session":  "object",
}

# ── Localización de CSVs ──────────────────────────────────────────────────────

def _find_csv_files() -> list:
    paths = []
    for fname in config.CSV_FILES:
        candidate = os.path.join(config.DATA_DIR, fname)
        if os.path.exists(candidate):
            paths.append(candidate)

    if not paths:
        pattern = os.path.join(config.DATA_DIR, "**", "*.csv")
        paths   = sorted(glob.glob(pattern, recursive=True))

    if not paths:
        raise FileNotFoundError(
            f"No se encontraron CSVs en {config.DATA_DIR}.\n"
            "Ejecuta: python -c \"import kagglehub; "
            "kagglehub.dataset_download('mkechinov/ecommerce-behavior-data-from-multi-category-store')\""
        )

    log.info("Archivos CSV encontrados: %d", len(paths))
    for p in paths:
        log.info("  %s  (%.2f GB)", os.path.basename(p), os.path.getsize(p) / 1e9)
    return paths


# ── Transformación por partición (pandas puro, rápido) ───────────────────────

def _transform_partition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe una partición ~128 MB y devuelve el DataFrame limpio y enriquecido.
    Toda la lógica es pandas nativo → sin overhead de Dask types.
    """
    # Filtros básicos
    df = df[df["price"] > 0].copy()
    df = df.dropna(subset=["user_id"])
    if df.empty:
        return df

    # Rellenar nulos de strings sin pasar por la lógica de categorías de pandas
    df["brand"]         = df["brand"].astype("object").fillna("unknown").astype(str)
    df["category_code"] = df["category_code"].astype("object").fillna("unknown").astype(str)

    # Componentes temporales
    et = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df["year"]  = et.dt.year.astype("int16")
    df["month"] = et.dt.month.astype("int8")
    df["hour"]  = et.dt.hour.astype("int8")
    df["dow"]   = et.dt.dayofweek.astype("int8")

    # Categoría top-level
    df["category_top"] = df["category_code"].str.split(".", n=1).str[0]

    # Bucket de precio
    df["price_bucket"] = pd.cut(
        df["price"],
        bins=[-np.inf, 50, 300, np.inf],
        labels=["low", "mid", "high"],
    ).astype(str)

    # Eliminar columnas innecesarias
    df = df.drop(
        columns=["user_session", "event_time", "category_id",
                 "event_type", "category_code"],
        errors="ignore",
    )
    return df


# ── Aggregations incrementales ────────────────────────────────────────────────

class _IncrementalAggregator:
    """
    Acumula estadísticas partición a partición sin retener datos originales.
    """
    def __init__(self):
        self.brand_revenue  = defaultdict(float)
        self.cat_count      = defaultdict(int)
        self.hourly_count   = defaultdict(int)
        self.hourly_sum     = defaultdict(float)
        self.monthly_count  = defaultdict(int)
        self.monthly_sum    = defaultdict(float)
        self.bucket_count   = defaultdict(int)
        self.bucket_sum     = defaultdict(float)
        self.bucket_sum2    = defaultdict(float)
        self.bucket_min     = {}
        self.bucket_max     = {}
        self.total_events   = 0
        self.total_revenue  = 0.0
        # Conjuntos de IDs — crecen con la cardinalidad, no con el volumen
        self.unique_users    = set()
        self.unique_products = set()

    def update(self, df: pd.DataFrame):
        if df.empty:
            return
        self.total_events  += len(df)
        self.total_revenue += float(df["price"].sum())
        self.unique_users.update(df["user_id"].tolist())
        self.unique_products.update(df["product_id"].tolist())

        for brand, rev in df.groupby("brand")["price"].sum().items():
            self.brand_revenue[str(brand)] += float(rev)

        for cat, cnt in df["category_top"].value_counts().items():
            self.cat_count[str(cat)] += int(cnt)

        for hour, grp in df.groupby("hour")["price"]:
            h = int(hour)
            self.hourly_count[h] += len(grp)
            self.hourly_sum[h]   += float(grp.sum())

        for (yr, mo), grp in df.groupby(["year", "month"])["price"]:
            k = (int(yr), int(mo))
            self.monthly_count[k] += len(grp)
            self.monthly_sum[k]   += float(grp.sum())

        for bucket, grp in df.groupby("price_bucket")["price"]:
            b    = str(bucket)
            vals = grp.values.astype(np.float64)
            self.bucket_count[b] += len(vals)
            self.bucket_sum[b]   += float(vals.sum())
            self.bucket_sum2[b]  += float((vals ** 2).sum())
            self.bucket_min[b]    = min(self.bucket_min.get(b, np.inf), float(vals.min()))
            self.bucket_max[b]    = max(self.bucket_max.get(b, -np.inf), float(vals.max()))

    def to_metrics(self) -> dict:
        top_brands = dict(sorted(self.brand_revenue.items(), key=lambda x: -x[1])[:30])
        top_cats   = dict(sorted(self.cat_count.items(), key=lambda x: -x[1])[:20])

        hourly = []
        for h in range(24):
            cnt  = self.hourly_count.get(h, 0)
            s    = self.hourly_sum.get(h, 0.0)
            hourly.append({"hour": h, "count": cnt, "sum": s,
                           "mean": s / cnt if cnt else 0.0})

        monthly = []
        for (yr, mo) in sorted(self.monthly_count):
            cnt = self.monthly_count[(yr, mo)]
            s   = self.monthly_sum[(yr, mo)]
            monthly.append({"year": yr, "month": mo, "count": cnt, "sum": s,
                             "mean": s / cnt if cnt else 0.0})

        price_bucket_stats = []
        for b in ["low", "mid", "high"]:
            cnt  = self.bucket_count.get(b, 0)
            s    = self.bucket_sum.get(b, 0.0)
            s2   = self.bucket_sum2.get(b, 0.0)
            mean = s / cnt if cnt else 0.0
            var  = (s2 / cnt - mean ** 2) if cnt else 0.0
            price_bucket_stats.append({
                "price_bucket": b, "count": cnt,
                "mean": round(mean, 4),
                "std":  round(float(np.sqrt(max(var, 0))), 4),
                "min":  round(self.bucket_min.get(b, 0.0), 4),
                "max":  round(self.bucket_max.get(b, 0.0), 4),
            })

        return {
            "revenue_by_brand":   top_brands,
            "count_by_category":  top_cats,
            "hourly_stats":       hourly,
            "monthly_stats":      monthly,
            "price_bucket_stats": price_bucket_stats,
            "total_events":       self.total_events,
            "total_revenue":      round(self.total_revenue, 2),
            "unique_users":       len(self.unique_users),
            "unique_products":    len(self.unique_products),
        }


# ── Procesamiento de un CSV (unidad de trabajo) ───────────────────────────────

def _process_one_csv(csv_path: str, agg: _IncrementalAggregator,
                     parquet_dir: str, csv_index: int) -> None:
    """
    Lee UN CSV con Dask (scheduler=synchronous), procesa partición a partición
    y escribe un único archivo Parquet con PyArrow streaming writer.

    scheduler="synchronous" → sin workers distribuidos, sin copias extra en RAM.
    """
    fname = os.path.basename(csv_path)
    log.info("── Procesando %s …", fname)

    with dask.config.set(scheduler="synchronous"):
        ddf = dd.read_csv(
            csv_path,
            dtype=DTYPE_MAP,
            blocksize="128MB",      # ← particiones pequeñas
            assume_missing=True,
            low_memory=True,
        )

        # Schema de salida para el meta de map_partitions
        meta_dtypes = {
            "product_id":   "int32",
            "brand":        "object",
            "price":        "float32",
            "user_id":      "int32",
            "year":         "int16",
            "month":        "int8",
            "hour":         "int8",
            "dow":          "int8",
            "category_top": "object",
            "price_bucket": "object",
        }
        ddf_clean = ddf.map_partitions(_transform_partition, meta=meta_dtypes)

        n_parts = ddf_clean.npartitions
        log.info("  Particiones: %d (blocksize=128MB)", n_parts)

        out_dir = os.path.join(parquet_dir, f"file_{csv_index:02d}")
        os.makedirs(out_dir, exist_ok=True)
        pq_path = os.path.join(out_dir, "data.parquet")

        writer = None

        for i in range(n_parts):
            # Materializar SOLO esta partición
            part = ddf_clean.get_partition(i).compute()

            if part.empty:
                del part
                continue

            # Actualizar aggregations
            agg.update(part)

            # Escribir a Parquet (streaming — no acumula en RAM)
            table = pa.Table.from_pandas(part, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(pq_path, table.schema, compression="snappy")
            writer.write_table(table)

            # Liberar inmediatamente
            del part, table

            if (i + 1) % 10 == 0 or (i + 1) == n_parts:
                log.info("  [%s] Partición %d/%d", fname, i + 1, n_parts)

        if writer:
            writer.close()

    log.info("  ✓ %s completado", fname)


# ── Punto de entrada ──────────────────────────────────────────────────────────

def run() -> dict:
    t0 = time.perf_counter()
    os.makedirs(config.PARQUET_DIR, exist_ok=True)

    csv_paths = _find_csv_files()
    agg       = _IncrementalAggregator()

    for idx, csv_path in enumerate(csv_paths):
        _process_one_csv(csv_path, agg, config.PARQUET_DIR, idx)

    metrics = agg.to_metrics()

    with open(config.METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info("Métricas guardadas en %s", config.METRICS_FILE)

    elapsed = time.perf_counter() - t0
    log.info("FASE 1 completada en %.1f s", elapsed)
    log.info("  Eventos totales:  %s", f"{metrics['total_events']:,}")
    log.info("  Revenue total:    $%s", f"{metrics['total_revenue']:,.2f}")
    log.info("  Usuarios únicos:  %s", f"{metrics['unique_users']:,}")
    log.info("  Productos únicos: %s", f"{metrics['unique_products']:,}")

    metrics["phase1_elapsed_s"] = round(elapsed, 2)
    return metrics


if __name__ == "__main__":
    run()