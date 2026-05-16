# HyperCommerce-Pipeline

Proyecto de procesamiento por fases para un dataset de comportamiento e-commerce.

**Resumen**
- Pipeline en 4 fases que transforma raw CSVs a features por usuario y segmentaciones:
  1. Fase 1 — Ingesta out-of-core con Dask (CSV -> Parquet)
  2. Fase 2 — Extracción de features en CPU (multiprocessing)
  3. Fase 3 — Cómputo de similitud y segmentación en GPU (CUDA)
  4. Fase 4 — Dashboard con Streamlit

**Objetivo**: procesar un dataset grande (~5–6 GB por archivo) con memoria limitada, generar features por usuario y clusters/segmentos reproducibles.

---

**Contenido del repositorio**
- `pipeline_runner.py`: orquestador principal.
- `config.py`: configuración (rutas, tamaños de chunk, workers).
- `requirements.txt`: dependencias Python.
- `phase1_dask_ingestion/ingest.py`: ingesta out-of-core y escritura Parquet.
- `phase2_multiprocessing/cpu_features.py`: extracción paralela de features por usuario.
- `phase3_cuda/gpu_compute.py`, `phase3_cuda/cuda_kernels.cu`: GPU processing (opcional, requiere CUDA).
- `phase4_streamlit/dashboard.py`: visualización interactiva con Streamlit.
- `outputs/`: resultados (parquet_clean/, métricas, features, etc.).

---

**Dataset**
- Fuente: `mkechinov/ecommerce-behavior-data-from-multi-category-store` (Kaggle).
- URL (archivo ejemplo): https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store?select=2019-Nov.csv
- Descripción general: el archivo contiene datos de comportamiento del usuario durante 7 meses (octubre 2019 — abril 2020) de una gran tienda online multi-categoría. Cada fila representa un evento (relacionado a productos y usuarios).
- Formato original: CSV mensuales (~5–6 GB por archivo en los ejemplos usados).
- Propiedades / columnas relevantes (definición):
  - `event_time`: momento en que ocurrió el evento (UTC).
  - `event_type`: tipo de evento. En este dataset los eventos que nos interesan son de compra (`purchase`).
  - `product_id`: ID del producto.
  - `category_id`: ID numérico de la categoría del producto.
  - `category_code`: taxonomy string de la categoría cuando está disponible (p. ej. `electronics.audio`). A veces está ausente para accesorios u categorías no mapeables.
  - `brand`: cadena con el nombre de la marca en minúsculas; puede estar ausente.
  - `price`: precio del producto (float). Presente para eventos de compra.
  - `user_id`: identificador permanente del usuario.
  - `user_session`: identificador temporal de la sesión del usuario (cambia tras lapsos largos y es igual para eventos dentro de la misma sesión).
- Notas y consideraciones:
  - El dataset contiene eventos muchos-a-muchos entre usuarios y productos; un mismo `user_id` aparece en múltiples filas.
  - `category_code` y `brand` pueden contener nulos; durante la ingesta se rellenan con `unknown` y se convierten a `object` para evitar problemas con `Categorical` en pandas.
  - Las fechas están en UTC y se usan para generar features temporales (año, mes, hora, dow).
  - El pipeline asume que los CSVs están en `config.DATA_DIR` y puede descargar automáticamente el dataset con `kagglehub` si no se encuentran.

---

**Fase 1 — Dask Out-of-Core Ingestion**
- Propósito: leer CSVs grandes sin cargar todo en memoria, limpiar y escribir Parquet particionado por archivo fuente.
- Estrategias principales:
  - `dask.dataframe.read_csv(..., blocksize='128MB')` → particiones pequeñas.
  - `map_partitions` con transformaciones en pandas puro para minimizar overhead.
  - `pyarrow` ParquetWriter en streaming para no acumular en RAM.
- Salidas:
  - Parquet por archivo en `outputs/parquet_clean/file_XX/data.parquet`.
  - `outputs/metrics.json` con métricas incrementales (events, revenue, top brands, etc.).

**Errores comunes y soluciones** (Fase 1)
- `Cannot setitem on a Categorical with a new category (unknown)`: convertir columnas categóricas a `object` antes de `fillna("unknown")` (ya aplicado en `phase1_dask_ingestion/ingest.py`).

---

**Fase 2 — CPU Multiprocessing Features**
- Propósito: cargar Parquet (solo columnas necesarias) y calcular features por `user_id` en paralelo.
- Estrategias principales:
  - Lectura con `pyarrow` y particionado en chunks de `CHUNK_SIZE * CPU_WORKERS` filas.
  - `multiprocessing.Pool` con `imap_unordered` para distribuir trabajo.
  - Agregación final por `user_id` para combinar resultados parciales.
- Salida: `outputs/features_cpu.parquet` (por defecto `config.CPU_RESULT`).

**Errores comunes y soluciones** (Fase 2)
- `ParquetDataset(..., use_legacy_dataset=...)` fallaba en ciertas versiones de PyArrow; la carga ahora usa un `glob` recursivo y `pq.read_table` para compatibilidad.

---

**Fase 3 — GPU/CUDA Similarity & Segmentation**
- Propósito: construir matriz de similitud entre usuarios y ejecutar clustering/segmentación en GPU para acelerar operaciones densas (si dispone de CUDA).
- Requisitos opcionales: NVIDIA CUDA Toolkit, drivers compatibles, `numba`/`cupy`/`pycuda` o bindings personalizados según `phase3_cuda/gpu_compute.py`.

**Análisis de Rendimiento**

| Fase | Tiempo de ejecución | Modo |
| --- | ---: | --- |
| Fase 1 | 379.7 s | CPU / Dask |
| Fase 2 | 3626.1 s | CPU / Multiprocessing |
| Fase 3 | 2.9 s | GPU |

---

**Fase 4 — Dashboard**
- `phase4_streamlit/dashboard.py` — visualización y exploración de métricas y segmentos.
- Comando: `streamlit run phase4_streamlit/dashboard.py`.

---

**Instalación y requisitos**
1. Crear y activar virtualenv (ejemplo):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Requisitos opcionales:
 - CUDA (para Fase 3) y drivers NVIDIA.
 - Aceleradores BLAS/OpenBLAS para numpy/pandas si desea mejorar rendimiento.

---

**Cómo ejecutar el pipeline**
- Ejecutar todas las fases:

```bash
python pipeline_runner.py
```

- Ejecutar una fase específica:

```bash
python pipeline_runner.py --phase 1
python pipeline_runner.py --phase 2
python pipeline_runner.py --phase 3
```


**Requisitos y ejecución (detallado)**

1) **Requisitos mínimos**
- Python 3.8 o superior.
- Espacio en disco suficiente para los CSVs y Parquets (~x GB según los datos que descargues).

2) **Crear y activar entorno Python**
- En WSL / Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- En PowerShell (Windows):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3) **Ajustar `config.py`**
- Abra `config.py` y confirme las rutas: `DATA_DIR`, `OUTPUT_DIR`, `CPU_WORKERS`, `CHUNK_SIZE`, y `GPU_ENABLED` (si existe). Modifique según su máquina.

4) **Descargar el dataset (Kaggle)**
- Colocar el token `kaggle.json` en `~/.kaggle/kaggle.json` y proteger permisos:

```bash
mkdir -p ~/.kaggle
cp /ruta/a/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

- El `pipeline_runner.py` intentará descargar automáticamente si los CSVs no están en `config.DATA_DIR`. Si desea descargar manualmente, use la URL indicada en el README.

5) **Ejecutar fases del pipeline**
- Ejecutar todas las fases:

```bash
python pipeline_runner.py
```

- Ejecutar una fase concreta:

```bash
python pipeline_runner.py --phase 1
python pipeline_runner.py --phase 2
python pipeline_runner.py --phase 3
python pipeline_runner.py --phase 4
```

6) **Soporte GPU / CUDA (opcional)**
- Comprobar presencia de GPU y drivers:

```bash
nvidia-smi
```

- Instalar NVIDIA driver y CUDA Toolkit compatibles.
- Instalar paquetes Python GPU (ejemplo para CUDA 11.x):

```bash
pip install cupy-cuda11x
```

- Alternativa con conda (gestiona dependencias CUDA mejor):

```bash
conda create -n hpc python=3.10
conda activate hpc
conda install -c conda-forge cupy
pip install -r requirements.txt
```

- Probar GPU desde Python:

```bash
python - <<'PY'
import sys
try:
  import cupy as cp
  print('GPU OK, cupy version:', cp.__version__)
except Exception as e:
  print('GPU test failed:', e, file=sys.stderr)
PY
```

7) **Ejecutar el dashboard (Streamlit)**

```bash
streamlit run phase4_streamlit/dashboard.py
```

8) **Ajustes y debugging rápidos**
- Cambiar `blocksize` en `phase1_dask_ingestion/ingest.py` si tiene más RAM.
- Ajustar `config.CPU_WORKERS` y `CHUNK_SIZE` para la Fase 2 en `config.py`.
- Problemas con PyArrow/Parquet: verificar versión con `python -c "import pyarrow; print(pyarrow.__version__)"` y ajustar si es necesario.
- Mensajes sobre `Categorical` están resueltos en la ingesta (convierte a `object` y rellena `unknown`).

Notas de ejecución:
- Si no existen CSVs en `config.DATA_DIR`, el runner intenta descargar el dataset con `kagglehub`.
- La Fase 1 usa `dask` en modo `synchronous` por defecto (para bajo uso de RAM). Cambiar `dask.config` puede paralelizar si hay RAM suficiente.

---

**Outputs y ubicación**
- `outputs/metrics.json` — métricas agregadas por fase.
- `outputs/parquet_clean/` — Parquet generados por la Fase 1 (subcarpetas `file_00`, ...).
- `outputs/features_cpu.parquet` — features por usuario generadas en CPU (Fase 2).
- Otros artefactos temporales y logs se almacenan localmente.

---

**Consejos de rendimiento**
- Ajuste `blocksize` en `phase1_dask_ingestion/ingest.py` según RAM disponible (p. ej. 256MB si hay >16GB).
- Aumente `config.CPU_WORKERS` para usar más núcleos en Fase 2 (cuidado con I/O si disco es lento).
- Use NVMe o SSD para reducir I/O-bound en lecturas Parquet.

---

**Depuración y troubleshooting**
- Errores de tipos en pandas: inspeccione `DTYPE_MAP` en `phase1_dask_ingestion/ingest.py`.
- Si ve errores de PyArrow (API cambiada), consulte la versión instalada: `python -c "import pyarrow; print(pyarrow.__version__)"`.
- Mensaje `Cannot setitem on a Categorical with a new category`: solución aplicada convirtiendo primero a `object`.
- Mensaje `ParquetDataset(..., use_legacy_dataset=...)`: solución aplicada usando `glob` y `pq.read_table`.

---

**Resultados esperados**
- `metrics.json` con conteos y revenue total.
- `features_cpu.parquet` con una fila por `user_id` y ~10 features (frequency, total_spend, avg_price, entropy, top_brand, peak_hour, etc.).
- Segmentaciones/embeddings generados en Fase 3 si GPU disponible.

---

