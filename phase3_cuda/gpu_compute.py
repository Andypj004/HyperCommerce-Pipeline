"""
phase3_cuda/gpu_compute.py
━━━━━━━━━━━━━━━━━━━━━━━━━━
FASE 3 — Orquestador Python → CUDA (NumPy + ctypes)
─────────────────────────────────────────────────────
Flujo:
  1. Leer features de usuarios del Parquet de la Fase 2.
  2. Normalizar y estructurar como matriz NumPy float32.
  3. Segmentar usuarios en K clústeres simples (K-Means en CPU, ligero).
  4. Llamar a run_cosine_similarity() del .so CUDA:
       GPU calcula la matriz de similitud de coseno [N×N].
  5. Llamar a run_segment_stats() del .so CUDA:
       GPU calcula mean/std/min/max de cada feature por segmento.
  6. Guardar resultados comprimidos en .npz para la Fase 4.

Si no hay GPU disponible → fallback NumPy puro (más lento, mismo resultado).
"""

import os
import sys
import ctypes
import time
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans   # Ligero, rápido en CPU

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FASE-3] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Features numéricas que se envían a la GPU
NUMERIC_FEATURES = [
    "frequency", "total_spend", "avg_price", "std_price",
    "entropy_cat", "brand_loyalty", "weekend_ratio", "peak_hour",
]
N_SEGMENTS = 5   # Número de clústeres de usuarios


# ── Carga del .so CUDA ────────────────────────────────────────────────────────

def _load_cuda_lib() -> ctypes.CDLL | None:
    """Intenta cargar la biblioteca CUDA compilada. Devuelve None si falla."""
    so_path = config.CUDA_SO_PATH
    if not os.path.exists(so_path):
        log.warning("cuda_kernels.so no encontrado en %s. "
                    "Ejecuta: cd phase3_cuda && bash build_cuda.sh", so_path)
        return None
    try:
        lib = ctypes.CDLL(so_path)

        # Declarar tipos de retorno y argumentos para cada función exportada
        lib.cuda_device_info.restype = ctypes.c_longlong
        lib.cuda_device_info.argtypes = []

        lib.run_cosine_similarity.restype  = None
        lib.run_cosine_similarity.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # h_mat
            ctypes.POINTER(ctypes.c_float),  # h_sim
            ctypes.c_int,                    # N
            ctypes.c_int,                    # F
        ]

        lib.run_segment_stats.restype  = None
        lib.run_segment_stats.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # h_mat
            ctypes.POINTER(ctypes.c_int),    # h_seg_ids
            ctypes.POINTER(ctypes.c_float),  # h_mean
            ctypes.POINTER(ctypes.c_float),  # h_std
            ctypes.POINTER(ctypes.c_float),  # h_min
            ctypes.POINTER(ctypes.c_float),  # h_max
            ctypes.POINTER(ctypes.c_int),    # h_count
            ctypes.c_int,                    # N
            ctypes.c_int,                    # F
            ctypes.c_int,                    # S
        ]

        log.info("CUDA library cargada desde %s", so_path)
        vram = lib.cuda_device_info()
        log.info("VRAM disponible: %.2f GB", vram / 1e9)
        return lib

    except OSError as e:
        log.warning("No se pudo cargar CUDA lib: %s → usando fallback NumPy", e)
        return None


# ── Preparación de datos ─────────────────────────────────────────────────────

def _prepare_matrix(features_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Extrae features numéricas, escala con StandardScaler y devuelve
    la matriz float32 lista para la GPU.
    """
    df = features_df[NUMERIC_FEATURES].copy()

    # Rellenar NaN con mediana de columna
    for col in df.columns:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    # Limitar usuarios enviados a GPU según VRAM disponible
    # (Matriz NxN de float32: N=2048 → 2048²×4 = ~16 MB; N=8192 → ~256 MB)
    N_max = config.GPU_BATCH_ROWS
    if len(df) > N_max:
        log.info("Submuestreo a %d usuarios para GPU (total: %d)", N_max, len(df))
        df = df.sample(N_max, random_state=42)
        user_ids = features_df.loc[df.index, "user_id"].tolist()
    else:
        user_ids = features_df["user_id"].tolist()

    scaler = StandardScaler()
    mat    = scaler.fit_transform(df).astype(np.float32)

    log.info("Matriz de features: %d usuarios × %d features", *mat.shape)
    return mat, scaler, user_ids


def _cluster_users(mat: np.ndarray) -> np.ndarray:
    """K-Means en CPU para asignar segmento a cada usuario."""
    log.info("Clustering K-Means (K=%d) en CPU …", N_SEGMENTS)
    km = MiniBatchKMeans(n_clusters=N_SEGMENTS, random_state=42,
                         batch_size=1024, max_iter=100)
    labels = km.fit_predict(mat).astype(np.int32)
    for s in range(N_SEGMENTS):
        log.info("  Segmento %d: %d usuarios", s, (labels == s).sum())
    return labels


# ── Operaciones GPU / fallback NumPy ─────────────────────────────────────────

def _cosine_sim_gpu(lib: ctypes.CDLL, mat: np.ndarray) -> np.ndarray:
    """Llama al kernel CUDA para calcular la matriz de similitud coseno."""
    N, F = mat.shape
    h_mat = mat.flatten().astype(np.float32)
    h_sim = np.zeros(N * N, dtype=np.float32)

    lib.run_cosine_similarity(
        h_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        h_sim.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(N),
        ctypes.c_int(F),
    )
    return h_sim.reshape(N, N)


def _cosine_sim_numpy(mat: np.ndarray) -> np.ndarray:
    """Fallback: similitud coseno con NumPy puro (más lento)."""
    log.info("Calculando similitud coseno con NumPy (fallback) …")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    mat_n = mat / norms
    return mat_n @ mat_n.T


def _segment_stats_gpu(
    lib: ctypes.CDLL, mat: np.ndarray, seg_ids: np.ndarray
) -> dict[str, np.ndarray]:
    """Llama al kernel CUDA para calcular estadísticas por segmento."""
    N, F = mat.shape
    S    = N_SEGMENTS

    h_mat  = mat.flatten().astype(np.float32)
    h_segs = seg_ids.astype(np.int32)
    h_mean = np.zeros(S * F, dtype=np.float32)
    h_std  = np.zeros(S * F, dtype=np.float32)
    h_min  = np.zeros(S * F, dtype=np.float32)
    h_max  = np.zeros(S * F, dtype=np.float32)
    h_cnt  = np.zeros(S, dtype=np.int32)

    lib.run_segment_stats(
        h_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        h_segs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        h_mean.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        h_std.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        h_min.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        h_max.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        h_cnt.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(N), ctypes.c_int(F), ctypes.c_int(S),
    )

    return {
        "mean":  h_mean.reshape(S, F),
        "std":   h_std.reshape(S, F),
        "min":   h_min.reshape(S, F),
        "max":   h_max.reshape(S, F),
        "count": h_cnt,
    }


def _segment_stats_numpy(mat: np.ndarray, seg_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Fallback NumPy para estadísticas por segmento."""
    S, F = N_SEGMENTS, mat.shape[1]
    stats = {k: np.zeros((S, F), dtype=np.float32) for k in ["mean","std","min","max"]}
    stats["count"] = np.zeros(S, dtype=np.int32)

    for s in range(S):
        mask = seg_ids == s
        if mask.sum() == 0:
            continue
        sub = mat[mask]
        stats["mean"][s]  = sub.mean(axis=0)
        stats["std"][s]   = sub.std(axis=0)
        stats["min"][s]   = sub.min(axis=0)
        stats["max"][s]   = sub.max(axis=0)
        stats["count"][s] = mask.sum()

    return stats


# ── Punto de entrada ─────────────────────────────────────────────────────────

def run() -> dict:
    """
    Ejecuta la Fase 3 completa.
    Devuelve un dict con arrays NumPy de los resultados.
    """
    t0 = time.perf_counter()

    # Leer features de Fase 2
    log.info("Leyendo features de usuario desde %s", config.CPU_RESULT)
    features_df = pd.read_parquet(config.CPU_RESULT)
    log.info("Usuarios cargados: %d", len(features_df))

    # Preparar matriz NumPy
    mat, scaler, user_ids = _prepare_matrix(features_df)

    # Clustering CPU
    seg_ids = _cluster_users(mat)

    # Cargar CUDA lib
    lib = _load_cuda_lib()
    gpu_mode = lib is not None

    # Similitud coseno
    log.info("Calculando matriz de similitud coseno (%s) …",
             "CUDA GPU" if gpu_mode else "NumPy fallback")
    t_sim = time.perf_counter()
    if gpu_mode:
        sim_matrix = _cosine_sim_gpu(lib, mat)
    else:
        sim_matrix = _cosine_sim_numpy(mat)
    log.info("  → Similitud coseno: %.2f s", time.perf_counter() - t_sim)

    # Estadísticas por segmento
    log.info("Calculando estadísticas por segmento (%s) …",
             "CUDA GPU" if gpu_mode else "NumPy fallback")
    t_seg = time.perf_counter()
    if gpu_mode:
        seg_stats = _segment_stats_gpu(lib, mat, seg_ids)
    else:
        seg_stats = _segment_stats_numpy(mat, seg_ids)
    log.info("  → Estadísticas segmento: %.2f s", time.perf_counter() - t_seg)

    # Guardar resultados
    output = {
        "sim_matrix":   sim_matrix.astype(np.float32),
        "seg_ids":      seg_ids.astype(np.int32),
        "seg_mean":     seg_stats["mean"].astype(np.float32),
        "seg_std":      seg_stats["std"].astype(np.float32),
        "seg_min":      seg_stats["min"].astype(np.float32),
        "seg_max":      seg_stats["max"].astype(np.float32),
        "seg_count":    seg_stats["count"].astype(np.int32),
        "feature_names": np.array(NUMERIC_FEATURES),
        "user_ids":     np.array(user_ids[:len(seg_ids)]),
        "gpu_mode":     np.array([int(gpu_mode)]),
    }

    np.savez_compressed(config.GPU_RESULT, **output)
    log.info("Resultados GPU guardados en %s", config.GPU_RESULT)

    elapsed = time.perf_counter() - t0
    log.info("FASE 3 completada en %.1f s (modo: %s)",
             elapsed, "GPU" if gpu_mode else "CPU-fallback")

    return {
        "phase3_elapsed_s": round(elapsed, 2),
        "gpu_mode":         gpu_mode,
        "n_users_gpu":      len(user_ids),
        "sim_shape":        list(sim_matrix.shape),
        "n_segments":       N_SEGMENTS,
        "feature_names":    NUMERIC_FEATURES,
    }


if __name__ == "__main__":
    result = run()
    for k, v in result.items():
        print(f"  {k}: {v}")
