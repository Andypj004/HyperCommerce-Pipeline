"""
config.py — Configuración central del pipeline
Ajusta DATA_DIR y los parámetros de rendimiento según tu máquina.
"""
import os
import multiprocessing
import psutil


# ── Localización automática del dataset (cualquier versión kagglehub) ────────
def _find_kagglehub_data_dir() -> str:
    """
    Busca la versión más reciente descargada por kagglehub.
    Funciona con v7, v8, v9, etc.
    """
    base = os.path.expanduser(
        "~/.cache/kagglehub/datasets/"
        "mkechinov/ecommerce-behavior-data-from-multi-category-store/versions"
    )
    if os.path.isdir(base):
        versions = sorted(
            [d for d in os.listdir(base)
             if os.path.isdir(os.path.join(base, d))],
            key=lambda x: int(x) if x.isdigit() else 0,
        )
        if versions:
            return os.path.join(base, versions[-1])
    return base

DATA_DIR = os.environ.get("ECOM_DATA_DIR", _find_kagglehub_data_dir())

# Archivos CSV disponibles (se usa cualquiera que exista en DATA_DIR)
CSV_FILES = [
    "2019-Oct.csv",
    "2019-Nov.csv",
    "2019-Dec.csv",
    "2020-Jan.csv",
    "2020-Feb.csv",
    "2020-Mar.csv",
    "2020-Apr.csv",
]

# ── Directorios de salida ────────────────────────────────────────────────────
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARQUET_DIR  = os.path.join(OUTPUT_DIR, "parquet_clean")
CPU_RESULT   = os.path.join(OUTPUT_DIR, "cpu_features.parquet")
GPU_RESULT   = os.path.join(OUTPUT_DIR, "gpu_results.npz")
METRICS_FILE = os.path.join(OUTPUT_DIR, "metrics.json")

# ── Dask ─────────────────────────────────────────────────────────────────────
# Modo synchronous → sin workers distribuidos → RAM mínima
# (No se usan DASK_WORKERS/DASK_THREADS con scheduler=synchronous)
TOTAL_RAM_GB   = psutil.virtual_memory().total / 1e9
DASK_WORKERS   = 1     # Reservado para compatibilidad; no se usa con synchronous
DASK_THREADS   = 1

# ── Multiprocessing ──────────────────────────────────────────────────────────
# Para Fase 2 dejamos n-1 cores libres al SO; mínimo 1
CPU_WORKERS = max(1, multiprocessing.cpu_count() - 1)
CHUNK_SIZE  = 30_000   # Filas por chunk (reducido respecto al original)

# ── CUDA / GPU ───────────────────────────────────────────────────────────────
CUDA_SO_PATH = os.path.join(os.path.dirname(__file__),
                            "phase3_cuda", "cuda_kernels.so")

# Usuarios enviados a GPU en la Fase 3 (ajusta según VRAM disponible)
# 2048 → ~16 MB en GPU;  4096 → ~64 MB;  8192 → ~256 MB
GPU_BATCH_ROWS = 2_048

# ── Streamlit ────────────────────────────────────────────────────────────────
APP_TITLE = "E-Commerce Behavior — Pipeline Híbrido CPU·GPU"