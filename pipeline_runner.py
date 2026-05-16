"""
pipeline_runner.py
━━━━━━━━━━━━━━━━━━
Orquestador principal — ejecuta las 4 fases en secuencia y reporta métricas.

Uso:
  python pipeline_runner.py              # Ejecuta todas las fases
  python pipeline_runner.py --phase 1   # Solo Fase 1
  python pipeline_runner.py --phase 2   # Solo Fase 2 (requiere Fase 1)
  python pipeline_runner.py --phase 3   # Solo Fase 3 (requiere Fase 2)
"""

import os
import sys
import json
import time
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PIPELINE] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)
import config


def run_phase1():
    log.info("=" * 60)
    log.info("INICIANDO FASE 1 — Dask Out-of-Core Ingestion")
    log.info("=" * 60)
    from phase1_dask_ingestion.ingest import run
    return run()


def run_phase2():
    log.info("=" * 60)
    log.info("INICIANDO FASE 2 — CPU Multiprocessing Features")
    log.info("=" * 60)
    from phase2_multiprocessing.cpu_features import run
    return run()


def run_phase3():
    log.info("=" * 60)
    log.info("INICIANDO FASE 3 — GPU/CUDA Similarity & Segmentation")
    log.info("=" * 60)
    from phase3_cuda.gpu_compute import run
    return run()


def update_metrics(new_data: dict):
    """Fusiona métricas adicionales en el archivo JSON."""
    try:
        if os.path.exists(config.METRICS_FILE):
            with open(config.METRICS_FILE) as f:
                metrics = json.load(f)
        else:
            metrics = {}
        metrics.update({k: v for k, v in new_data.items()
                        if not isinstance(v, dict)})
        with open(config.METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
    except Exception as e:
        log.warning("No se pudo actualizar métricas: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Pipeline E-Commerce Híbrido")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3],
                        help="Ejecutar solo una fase específica")
    args = parser.parse_args()

    t_total = time.perf_counter()
    summary  = {}

    # ── Descarga del dataset (si no existe) ──────────────────────────────────
    if not os.path.exists(config.DATA_DIR) or not any(
        f.endswith(".csv") for f in os.listdir(config.DATA_DIR)
        if os.path.isfile(os.path.join(config.DATA_DIR, f))
    ):
        log.info("Dataset no encontrado en %s. Descargando con kagglehub …",
                 config.DATA_DIR)
        try:
            import kagglehub
            path = kagglehub.dataset_download(
                "mkechinov/ecommerce-behavior-data-from-multi-category-store"
            )
            log.info("Dataset descargado en: %s", path)
            # Actualizar DATA_DIR con la ruta real
            config.DATA_DIR = path
        except Exception as e:
            log.error("Error al descargar dataset: %s", e)
            log.error("Descarga manual: https://www.kaggle.com/datasets/"
                      "mkechinov/ecommerce-behavior-data-from-multi-category-store")
            sys.exit(1)

    # ── Ejecución de fases ────────────────────────────────────────────────────
    try:
        if args.phase is None or args.phase == 1:
            m1 = run_phase1()
            summary["fase1"] = m1
            update_metrics(m1)

        if args.phase is None or args.phase == 2:
            m2 = run_phase2()
            summary["fase2"] = {"phase2_elapsed_s": getattr(m2, "attrs", {})
                                 .get("phase2_elapsed_s", "?")}
            update_metrics(summary["fase2"])

        if args.phase is None or args.phase == 3:
            m3 = run_phase3()
            summary["fase3"] = m3
            update_metrics(m3)

    except KeyboardInterrupt:
        log.warning("Pipeline interrumpido por el usuario.")
        sys.exit(0)
    except Exception as e:
        log.exception("Error en el pipeline: %s", e)
        sys.exit(1)

    # ── Resumen ───────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_total
    log.info("=" * 60)
    log.info("PIPELINE COMPLETADO en %.1f s", elapsed)
    log.info("=" * 60)

    for fase, meta in summary.items():
        log.info("  %s: %s", fase, {k: v for k, v in meta.items()
                                    if isinstance(v, (int, float, str, bool))})

    log.info("")
    log.info("Lanzar dashboard:")
    log.info("  streamlit run phase4_streamlit/dashboard.py")


if __name__ == "__main__":
    main()
