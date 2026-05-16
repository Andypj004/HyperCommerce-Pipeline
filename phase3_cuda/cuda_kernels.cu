/*
 * phase3_cuda/cuda_kernels.cu
 * ═══════════════════════════════════════════════════════════════════════════
 * FASE 3 — Kernel CUDA en C++ para aceleración GPU
 * ─────────────────────────────────────────────────────────────────────────
 *
 * OPERACIÓN: Cálculo de Matriz de Similitud de Coseno entre perfiles de
 *            usuario (user feature vectors) con normalización L2 en GPU.
 *
 * ¿Por qué esta operación?
 *   - Con N usuarios × F features, la matriz de similitud NxN tiene complejidad
 *     O(N²·F). Para N=100 000 y F=8, eso es 80 000 000 000 operaciones FP.
 *   - En CPU (secuencial) tardaría minutos. En GPU con miles de hilos: segundos.
 *   - La similitud de coseno entre perfiles de usuario es el núcleo de sistemas
 *     de recomendación colaborativa (collaborative filtering).
 *
 * KERNELS implementados:
 *   1. normalize_rows_kernel  — Normalización L2 de cada fila (cada usuario).
 *   2. cosine_similarity_kernel — Producto punto entre filas normalizadas
 *                                  = similitud coseno ∈ [-1, 1].
 *   3. segment_stats_kernel   — Estadísticas (mean, std, min, max) de cada
 *                               columna de features por segmento de usuario.
 *
 * Compilar con:
 *   nvcc -O3 -shared -fPIC -o cuda_kernels.so cuda_kernels.cu
 *
 * La interfaz Python usa ctypes para llamar a las funciones exportadas.
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ── Macros de comprobación de errores CUDA ─────────────────────────────── */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d  '%s'\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)


/* ══════════════════════════════════════════════════════════════════════════
 * KERNEL 1: Normalización L2 de filas
 *   Cada hilo procesa UNA fila (un usuario).
 *   Calcula la norma L2 y divide cada elemento de la fila por ella.
 * ══════════════════════════════════════════════════════════════════════════ */
__global__ void normalize_rows_kernel(float* mat, int nrows, int ncols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    /* Calcular norma L2 de la fila */
    float norm_sq = 0.0f;
    for (int c = 0; c < ncols; c++) {
        float v = mat[row * ncols + c];
        norm_sq += v * v;
    }
    float inv_norm = (norm_sq > 1e-10f) ? rsqrtf(norm_sq) : 0.0f;

    /* Normalizar in-place */
    for (int c = 0; c < ncols; c++) {
        mat[row * ncols + c] *= inv_norm;
    }
}


/* ══════════════════════════════════════════════════════════════════════════
 * KERNEL 2: Similitud de Coseno entre todas las pares de filas (NxN)
 *   Grid 2D: cada hilo (i, j) calcula sim[i][j] = dot(row_i, row_j)
 *   sobre la matriz ya normalizada → resultado es cosine similarity.
 *
 *   Para matrices grandes, se usa tiling con shared memory para
 *   minimizar accesos a memoria global (coalesced access pattern).
 *
 *   TILE_SIZE: número de columnas procesadas por tile.
 * ══════════════════════════════════════════════════════════════════════════ */
#define TILE_SIZE 16

__global__ void cosine_similarity_kernel(
    const float* __restrict__ mat,   /* Matriz normalizada [N x F] */
    float*       __restrict__ sim,   /* Salida: matriz similitud [N x N] */
    int N,                           /* Número de filas (usuarios)        */
    int F                            /* Número de features                */
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  /* usuario i */
    int col = blockIdx.x * blockDim.x + threadIdx.x;  /* usuario j */

    if (row >= N || col >= N) return;

    /* Producto punto entre row i y row j */
    float dot = 0.0f;

    /* Iteración por tiles de F */
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int num_tiles = (F + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int f_a = t * TILE_SIZE + threadIdx.x;
        int f_b = t * TILE_SIZE + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] = (row < N && f_a < F) ? mat[row * F + f_a] : 0.0f;
        tile_B[threadIdx.x][threadIdx.y] = (col < N && f_b < F) ? mat[col * F + f_b] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            dot += tile_A[threadIdx.y][k] * tile_B[threadIdx.x][k];
        }
        __syncthreads();
    }

    /* Clamp a [-1, 1] por errores de redondeo FP32 */
    dot = fminf(1.0f, fmaxf(-1.0f, dot));
    sim[row * N + col] = dot;
}


/* ══════════════════════════════════════════════════════════════════════════
 * KERNEL 3: Estadísticas por segmento (mean, std, min, max por columna)
 *   Asignamos un bloque por segmento y procesamos features en paralelo.
 *   Cada hilo en el bloque maneja una feature distinta.
 * ══════════════════════════════════════════════════════════════════════════ */
__global__ void segment_stats_kernel(
    const float* __restrict__ mat,       /* [N x F] features de usuarios     */
    const int*   __restrict__ seg_ids,   /* [N] ID de segmento por usuario   */
    float*       __restrict__ seg_mean,  /* [S x F] salida: media            */
    float*       __restrict__ seg_std,   /* [S x F] salida: desviación std   */
    float*       __restrict__ seg_min,   /* [S x F] salida: mínimo           */
    float*       __restrict__ seg_max,   /* [S x F] salida: máximo           */
    int*         __restrict__ seg_count, /* [S] número de usuarios/segmento  */
    int N,                               /* Número de usuarios               */
    int F,                               /* Número de features               */
    int S                                /* Número de segmentos              */
) {
    int feat = blockIdx.x * blockDim.x + threadIdx.x;  /* feature index */
    if (feat >= F) return;

    /* Inicializar acumuladores locales por segmento */
    /* Para evitar atomics costosos, cada hilo acumula su propia feature */
    for (int s = 0; s < S; s++) {
        seg_mean [s * F + feat] = 0.0f;
        seg_std  [s * F + feat] = 0.0f;
        seg_min  [s * F + feat] =  1e30f;
        seg_max  [s * F + feat] = -1e30f;
    }

    /* Paso 1: calcular sumas y min/max */
    for (int i = 0; i < N; i++) {
        int s = seg_ids[i];
        if (s < 0 || s >= S) continue;
        float v = mat[i * F + feat];
        atomicAdd(&seg_mean[s * F + feat], v);
        atomicAdd(&seg_std [s * F + feat], v * v);  /* Suma de cuadrados */
        atomicMinf: /* placeholder — ver nota */ ;
    }
    /* Nota: atomicMin/Max no existe para float en CUDA < 9.
       Usamos atomicCAS con int reinterpretado — implementación completa abajo */
}

/* Versión completa con atomicMin/Max float via CAS */
__device__ float atomicMinFloat(float* addr, float val) {
    int* addr_i = (int*)addr;
    int  old_i  = *addr_i;
    int  val_i;
    do {
        val_i = old_i;
        float old_f = __int_as_float(old_i);
        if (old_f <= val) break;
        old_i = atomicCAS(addr_i, val_i, __float_as_int(val));
    } while (val_i != old_i);
    return __int_as_float(old_i);
}

__device__ float atomicMaxFloat(float* addr, float val) {
    int* addr_i = (int*)addr;
    int  old_i  = *addr_i;
    int  val_i;
    do {
        val_i = old_i;
        float old_f = __int_as_float(old_i);
        if (old_f >= val) break;
        old_i = atomicCAS(addr_i, val_i, __float_as_int(val));
    } while (val_i != old_i);
    return __int_as_float(old_i);
}

/* Kernel correcto con atomicMinFloat/MaxFloat */
__global__ void segment_stats_kernel_v2(
    const float* __restrict__ mat,
    const int*   __restrict__ seg_ids,
    float*       __restrict__ seg_sum,
    float*       __restrict__ seg_sum2,
    float*       __restrict__ seg_min,
    float*       __restrict__ seg_max,
    int*         __restrict__ seg_count,
    int N, int F, int S
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  /* usuario */
    if (i >= N) return;

    int s = seg_ids[i];
    if (s < 0 || s >= S) return;

    atomicAdd(&seg_count[s], 1);

    for (int f = 0; f < F; f++) {
        float v = mat[i * F + f];
        atomicAdd (&seg_sum [s * F + f], v);
        atomicAdd (&seg_sum2[s * F + f], v * v);
        atomicMinFloat(&seg_min[s * F + f], v);
        atomicMaxFloat(&seg_max[s * F + f], v);
    }
}

/* Kernel de postproceso: convertir sum/sum2/count → mean/std */
__global__ void finalize_stats_kernel(
    float* seg_sum, float* seg_sum2,
    float* seg_mean, float* seg_std,
    const int* seg_count,
    int S, int F
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * F;
    if (idx >= total) return;

    int s = idx / F;
    int f = idx % F;
    int cnt = seg_count[s];

    if (cnt == 0) {
        seg_mean[idx] = 0.0f;
        seg_std [idx] = 0.0f;
        return;
    }

    float mean = seg_sum[idx] / cnt;
    float var  = seg_sum2[idx] / cnt - mean * mean;
    seg_mean[idx] = mean;
    seg_std [idx] = sqrtf(fmaxf(0.0f, var));
}


/* ══════════════════════════════════════════════════════════════════════════
 * API C exportada — llamada desde Python vía ctypes
 * ══════════════════════════════════════════════════════════════════════════ */
extern "C" {

/*
 * run_cosine_similarity
 * ─────────────────────
 * Parámetros (host arrays, float32):
 *   h_mat   : [N x F] matriz de features por usuario (en RAM de CPU)
 *   h_sim   : [N x N] buffer de salida (en RAM de CPU, pre-alocado)
 *   N       : número de usuarios
 *   F       : número de features
 *
 * Flujo:
 *   1. Copiar h_mat → GPU (d_mat)
 *   2. Normalizar filas in-place (normalize_rows_kernel)
 *   3. Calcular similitudes (cosine_similarity_kernel)
 *   4. Copiar resultado GPU → h_sim
 */
void run_cosine_similarity(float* h_mat, float* h_sim, int N, int F) {
    float *d_mat = NULL, *d_sim = NULL;
    size_t mat_bytes = (size_t)N * F * sizeof(float);
    size_t sim_bytes = (size_t)N * N * sizeof(float);

    /* Alocar y transferir a GPU */
    CUDA_CHECK(cudaMalloc(&d_mat, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_sim, sim_bytes));
    CUDA_CHECK(cudaMemcpy(d_mat, h_mat, mat_bytes, cudaMemcpyHostToDevice));

    /* Kernel 1: Normalización — 1 hilo por usuario */
    int blk1 = 256;
    int grd1 = (N + blk1 - 1) / blk1;
    normalize_rows_kernel<<<grd1, blk1>>>(d_mat, N, F);
    CUDA_CHECK(cudaGetLastError());

    /* Kernel 2: Similitud coseno — grid 2D TILE_SIZE x TILE_SIZE */
    dim3 blk2(TILE_SIZE, TILE_SIZE);
    dim3 grd2((N + TILE_SIZE - 1) / TILE_SIZE,
              (N + TILE_SIZE - 1) / TILE_SIZE);
    cosine_similarity_kernel<<<grd2, blk2>>>(d_mat, d_sim, N, F);
    CUDA_CHECK(cudaGetLastError());

    /* Copiar resultado al host */
    CUDA_CHECK(cudaMemcpy(h_sim, d_sim, sim_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_mat);
    cudaFree(d_sim);
    CUDA_CHECK(cudaDeviceSynchronize());
}


/*
 * run_segment_stats
 * ─────────────────
 * Calcula estadísticas (mean, std, min, max) de features por segmento.
 *
 * Parámetros (host arrays):
 *   h_mat      : [N x F] float32 — features normalizadas
 *   h_seg_ids  : [N]     int32   — segmento de cada usuario (0..S-1)
 *   h_mean     : [S x F] float32 — salida: media por segmento×feature
 *   h_std      : [S x F] float32 — salida: desviación estándar
 *   h_min      : [S x F] float32 — salida: mínimo
 *   h_max      : [S x F] float32 — salida: máximo
 *   h_count    : [S]     int32   — salida: usuarios por segmento
 *   N, F, S    : dimensiones
 */
void run_segment_stats(
    float* h_mat, int* h_seg_ids,
    float* h_mean, float* h_std, float* h_min, float* h_max, int* h_count,
    int N, int F, int S
) {
    float *d_mat=NULL, *d_sum=NULL, *d_sum2=NULL, *d_min=NULL, *d_max=NULL;
    float *d_mean=NULL, *d_std_out=NULL;
    int   *d_seg=NULL, *d_cnt=NULL;

    size_t mat_bytes = (size_t)N * F * sizeof(float);
    size_t sf_bytes  = (size_t)S * F * sizeof(float);
    size_t s_bytes   = (size_t)S * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_mat,     mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_seg,     (size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum,     sf_bytes));
    CUDA_CHECK(cudaMalloc(&d_sum2,    sf_bytes));
    CUDA_CHECK(cudaMalloc(&d_min,     sf_bytes));
    CUDA_CHECK(cudaMalloc(&d_max,     sf_bytes));
    CUDA_CHECK(cudaMalloc(&d_mean,    sf_bytes));
    CUDA_CHECK(cudaMalloc(&d_std_out, sf_bytes));
    CUDA_CHECK(cudaMalloc(&d_cnt,     s_bytes));

    /* Inicializar acumuladores */
    CUDA_CHECK(cudaMemset(d_sum,  0, sf_bytes));
    CUDA_CHECK(cudaMemset(d_sum2, 0, sf_bytes));
    CUDA_CHECK(cudaMemset(d_cnt,  0, s_bytes));

    /* Inicializar min=+inf, max=-inf */
    float pos_inf =  1e30f;
    float neg_inf = -1e30f;
    for (int i = 0; i < S * F; i++) {
        cudaMemcpy(d_min + i, &pos_inf, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_max + i, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
    }

    CUDA_CHECK(cudaMemcpy(d_mat, h_mat, mat_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seg, h_seg_ids, (size_t)N*sizeof(int), cudaMemcpyHostToDevice));

    /* Kernel 3v2: acumulación atómica — 1 hilo por usuario */
    int blk3 = 256;
    int grd3 = (N + blk3 - 1) / blk3;
    segment_stats_kernel_v2<<<grd3, blk3>>>(
        d_mat, d_seg, d_sum, d_sum2, d_min, d_max, d_cnt, N, F, S
    );
    CUDA_CHECK(cudaGetLastError());

    /* Kernel finalización: sum→mean, sum2→std */
    int total = S * F;
    int blk4  = 256;
    int grd4  = (total + blk4 - 1) / blk4;
    finalize_stats_kernel<<<grd4, blk4>>>(
        d_sum, d_sum2, d_mean, d_std_out, d_cnt, S, F
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copiar resultados al host */
    CUDA_CHECK(cudaMemcpy(h_mean,  d_mean,    sf_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_std,   d_std_out, sf_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_min,   d_min,     sf_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max,   d_max,     sf_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_count, d_cnt,     s_bytes,  cudaMemcpyDeviceToHost));

    cudaFree(d_mat); cudaFree(d_seg); cudaFree(d_sum); cudaFree(d_sum2);
    cudaFree(d_min); cudaFree(d_max); cudaFree(d_mean);
    cudaFree(d_std_out); cudaFree(d_cnt);
}


/*
 * cuda_device_info
 * ─────────────────
 * Imprime info del dispositivo GPU y devuelve el total de VRAM en bytes.
 */
long long cuda_device_info(void) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s  |  VRAM: %.1f GB  |  SM: %d  |  CUDA: %d.%d\n",
           prop.name,
           (double)prop.totalGlobalMem / 1e9,
           prop.multiProcessorCount,
           prop.major, prop.minor);
    fflush(stdout);
    return (long long)prop.totalGlobalMem;
}

} /* extern "C" */
