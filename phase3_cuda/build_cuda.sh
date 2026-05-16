#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "Compilando cuda_kernels.cu a cuda_kernels.so en $ROOT_DIR"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc no encontrado en PATH. Instala el CUDA Toolkit y asegúrate de que nvcc esté disponible." >&2
  exit 2
fi

# Compilar con nvcc. Ajusta flags si tu toolchain requiere otros paths.
nvcc -O3 -shared -Xcompiler -fPIC -o cuda_kernels.so cuda_kernels.cu -lcudart

echo "Compilación completada: $(pwd)/cuda_kernels.so"
