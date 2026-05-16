"""
phase4_streamlit/dashboard.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FASE 4 — Dashboard interactivo con Streamlit + Plotly
─────────────────────────────────────────────────────
Ejecutar: streamlit run phase4_streamlit/dashboard.py

El dashboard lee los outputs de las fases anteriores:
  • metrics.json     → KPIs globales (Fase 1)
  • cpu_features.parquet → features por usuario (Fase 2)
  • gpu_results.npz  → similitud coseno, segmentos (Fase 3)

Secciones:
  1. KPIs globales (tarjetas)
  2. Revenue por marca (barras horizontales interactivas)
  3. Distribución horaria de compras (área + línea)
  4. Revenue mensual (línea temporal)
  5. Segmentación de usuarios GPU (scatter PCA 2D + radar)
  6. Similitud coseno entre usuarios (heatmap)
  7. Filtro en tiempo real de features por segmento
  8. Botón para lanzar el pipeline completo (async)
"""

import os
import sys
import json
import time
import subprocess
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paleta de colores por segmento ───────────────────────────────────────────
SEG_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
SEG_NAMES  = [f"Segmento {i}" for i in range(5)]


# ── Loaders con caché ────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_metrics() -> dict:
    if not os.path.exists(config.METRICS_FILE):
        return {}
    with open(config.METRICS_FILE) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_user_features() -> pd.DataFrame:
    if not os.path.exists(config.CPU_RESULT):
        return pd.DataFrame()
    return pd.read_parquet(config.CPU_RESULT)


@st.cache_data(ttl=300)
def load_gpu_results() -> dict:
    path = str(config.GPU_RESULT) + ".npz"
    if not os.path.exists(path):
        path = config.GPU_RESULT  # sin extensión
    if not os.path.exists(path):
        return {}
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(metrics: dict, gpu: dict) -> dict:
    st.sidebar.title("⚙️ Control del Pipeline")

    # Estado del pipeline
    st.sidebar.markdown("### Estado de Fases")
    phase_status = {
        "Fase 1 — Dask":         os.path.exists(config.PARQUET_DIR),
        "Fase 2 — Multiproc":    os.path.exists(config.CPU_RESULT),
        "Fase 3 — GPU/NumPy":    os.path.exists(config.GPU_RESULT) or
                                  os.path.exists(config.GPU_RESULT + ".npz"),
    }
    for phase, done in phase_status.items():
        icon = "✅" if done else "⏳"
        st.sidebar.markdown(f"{icon} {phase}")

    # Botón para lanzar pipeline
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Ejecutar Pipeline Completo", use_container_width=True):
        st.session_state["pipeline_running"] = True
        st.session_state["pipeline_log"]     = []

    # Filtros de análisis
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filtros")
    filters = {}

    if metrics:
        # Filtro de precio
        price_opts = ["Todos", "low", "mid", "high"]
        filters["price_bucket"] = st.sidebar.selectbox(
            "Bucket de Precio", price_opts, index=0
        )

        # Filtro de marca (top 20)
        brand_rev = metrics.get("revenue_by_brand", {})
        top_brands = ["Todas"] + list(brand_rev.keys())[:20]
        filters["brand"] = st.sidebar.selectbox("Marca", top_brands, index=0)

    if gpu:
        # Filtro de segmento
        n_seg = int(gpu.get("seg_count", np.array([5])).shape[0])
        seg_options = ["Todos"] + [f"Segmento {i}" for i in range(n_seg)]
        filters["segment"] = st.sidebar.selectbox("Segmento GPU", seg_options, index=0)

    return filters


# ── KPI Cards ────────────────────────────────────────────────────────────────

def render_kpis(metrics: dict):
    st.markdown("## 📊 KPIs Globales")
    c1, c2, c3, c4, c5 = st.columns(5)
    total_events  = metrics.get("total_events", 0)
    total_revenue = metrics.get("total_revenue", 0)
    unique_users  = metrics.get("unique_users", 0)
    unique_prods  = metrics.get("unique_products", 0)
    avg_ticket    = total_revenue / unique_users if unique_users else 0

    c1.metric("Total Eventos",    f"{total_events:,}")
    c2.metric("Revenue Total",    f"${total_revenue:,.0f}")
    c3.metric("Usuarios Únicos",  f"{unique_users:,}")
    c4.metric("Productos Únicos", f"{unique_prods:,}")
    c5.metric("Ticket Promedio",  f"${avg_ticket:,.2f}")


# ── Revenue por Marca ─────────────────────────────────────────────────────────

def render_brand_revenue(metrics: dict, filters: dict):
    st.markdown("## 🏷️ Revenue por Marca (Top 30)")
    brand_rev = metrics.get("revenue_by_brand", {})
    if not brand_rev:
        st.info("Sin datos de Fase 1. Ejecuta el pipeline primero.")
        return

    df = pd.DataFrame({"brand": list(brand_rev.keys()),
                        "revenue": list(brand_rev.values())})
    df = df.sort_values("revenue", ascending=True).tail(30)

    fig = px.bar(df, x="revenue", y="brand", orientation="h",
                 color="revenue", color_continuous_scale="Viridis",
                 labels={"revenue": "Revenue (USD)", "brand": "Marca"})
    fig.update_layout(height=500, coloraxis_showscale=False,
                      margin=dict(l=120, r=20, t=30, b=40))
    st.plotly_chart(fig, use_container_width=True)


# ── Distribución Horaria ──────────────────────────────────────────────────────

def render_hourly(metrics: dict):
    hourly = metrics.get("hourly_stats", [])
    if not hourly:
        return

    df = pd.DataFrame(hourly)
    df.columns = [c if c != "count" else "transacciones" for c in df.columns]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["hour"], y=df["transacciones"],
                         name="Transacciones", marker_color="#636EFA",
                         opacity=0.6), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["hour"], y=df["mean"],
                             name="Ticket Promedio (USD)", mode="lines+markers",
                             line=dict(color="#EF553B", width=2)),
                  secondary_y=True)
    fig.update_layout(title="Distribución Horaria de Compras",
                      xaxis_title="Hora del Día (UTC)",
                      height=350, legend=dict(x=0.01, y=0.99))
    fig.update_yaxes(title_text="Transacciones",   secondary_y=False)
    fig.update_yaxes(title_text="Ticket Promedio", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)


# ── Revenue Mensual ───────────────────────────────────────────────────────────

def render_monthly(metrics: dict):
    monthly = metrics.get("monthly_stats", [])
    if not monthly:
        return

    df = pd.DataFrame(monthly)
    df["periodo"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(df, x="periodo", y="sum", markers=True,
                      title="Revenue Mensual Total",
                      labels={"sum": "Revenue (USD)", "periodo": "Período"})
        fig.update_traces(line_color="#00CC96", line_width=2)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(df, x="periodo", y="count",
                     title="Transacciones por Mes",
                     labels={"count": "Transacciones", "periodo": "Período"},
                     color="count", color_continuous_scale="Blues")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


# ── Segmentación GPU ──────────────────────────────────────────────────────────

def render_segmentation(features_df: pd.DataFrame, gpu: dict, filters: dict):
    st.markdown("## 🤖 Segmentación de Usuarios (GPU / CUDA)")

    if features_df.empty or not gpu:
        st.info("Sin resultados de Fase 2/3. Ejecuta el pipeline primero.")
        return

    seg_ids = gpu.get("seg_ids", np.array([]))
    if len(seg_ids) == 0:
        return

    NUMERIC_FEATURES = [
        "frequency", "total_spend", "avg_price", "std_price",
        "entropy_cat", "brand_loyalty", "weekend_ratio", "peak_hour",
    ]

    # PCA 2D para visualización
    feat_cols = [c for c in NUMERIC_FEATURES if c in features_df.columns]
    sample_size = min(len(seg_ids), len(features_df))
    sub_df = features_df[feat_cols].iloc[:sample_size].fillna(0)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(sub_df.values)

    plot_df = pd.DataFrame({
        "PC1":      coords[:, 0],
        "PC2":      coords[:, 1],
        "Segmento": [f"Seg {s}" for s in seg_ids[:sample_size]],
        "frequency":    sub_df["frequency"].values if "frequency" in sub_df else 0,
        "total_spend":  sub_df["total_spend"].values if "total_spend" in sub_df else 0,
    })

    # Filtro por segmento
    seg_filter = filters.get("segment", "Todos")
    if seg_filter != "Todos":
        seg_num = int(seg_filter.split()[-1])
        plot_df = plot_df[plot_df["Segmento"] == f"Seg {seg_num}"]

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.scatter(
            plot_df, x="PC1", y="PC2", color="Segmento",
            size="frequency", hover_data=["total_spend"],
            color_discrete_sequence=SEG_COLORS,
            title="Mapa de Segmentos de Usuarios (PCA 2D)",
            labels={"PC1": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                    "PC2": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Radar de estadísticas por segmento
        seg_mean  = gpu.get("seg_mean",  None)
        feat_names = list(gpu.get("feature_names", np.array(NUMERIC_FEATURES)))
        if seg_mean is not None:
            n_seg = seg_mean.shape[0]
            fig_r = go.Figure()
            for s in range(n_seg):
                vals = list(seg_mean[s]) + [seg_mean[s][0]]
                cats = feat_names + [feat_names[0]]
                fig_r.add_trace(go.Scatterpolar(
                    r=vals, theta=cats, fill="toself",
                    name=f"Seg {s}", line_color=SEG_COLORS[s % len(SEG_COLORS)],
                    opacity=0.7,
                ))
            fig_r.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title="Perfil de Segmentos (Normalizado)",
                height=400, showlegend=True,
            )
            st.plotly_chart(fig_r, use_container_width=True)


# ── Heatmap de Similitud ──────────────────────────────────────────────────────

def render_similarity_heatmap(gpu: dict):
    st.markdown("## 🔥 Matriz de Similitud Coseno (GPU)")

    sim = gpu.get("sim_matrix", None)
    if sim is None:
        st.info("Sin resultados de Fase 3. Ejecuta el pipeline primero.")
        return

    # Mostrar submatriz (max 100×100 para rendimiento del navegador)
    n = min(100, sim.shape[0])
    sub = sim[:n, :n]

    fig = px.imshow(
        sub, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=f"Similitud Coseno entre usuarios (primeros {n}×{n})",
        labels={"color": "Similitud"},
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    gpu_mode = bool(int(gpu.get("gpu_mode", np.array([0]))[0]))
    st.caption(
        f"⚡ Calculado en {'**GPU (CUDA)**' if gpu_mode else '**CPU (NumPy fallback)**'}. "
        f"Dimensión completa: {sim.shape[0]}×{sim.shape[0]}"
    )


# ── Features por Segmento (filtro en tiempo real) ────────────────────────────

def render_feature_filter(features_df: pd.DataFrame, gpu: dict, filters: dict):
    st.markdown("## 🔍 Análisis de Features por Segmento (Tiempo Real)")

    if features_df.empty or not gpu:
        return

    seg_ids = gpu.get("seg_ids", np.array([]))
    if len(seg_ids) == 0:
        return

    NUMERIC_FEATURES = [
        "frequency", "total_spend", "avg_price", "std_price",
        "entropy_cat", "brand_loyalty", "weekend_ratio", "peak_hour",
    ]

    n_users = min(len(seg_ids), len(features_df))
    df = features_df.iloc[:n_users].copy()
    df["segmento"] = [f"Seg {s}" for s in seg_ids[:n_users]]

    # Filtro por segmento (tiempo real con Streamlit)
    seg_filter = filters.get("segment", "Todos")
    if seg_filter != "Todos":
        seg_num = int(seg_filter.split()[-1])
        df = df[df["segmento"] == f"Seg {seg_num}"]

    # Selector de feature a analizar
    feat_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    selected_feat = st.selectbox("Feature a visualizar", feat_cols, index=1)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df, x="segmento", y=selected_feat,
                     color="segmento", color_discrete_sequence=SEG_COLORS,
                     title=f"Distribución de '{selected_feat}' por Segmento",
                     points=False)
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x=selected_feat, color="segmento",
                           color_discrete_sequence=SEG_COLORS,
                           nbins=40, barmode="overlay",
                           title=f"Histograma de '{selected_feat}'",
                           opacity=0.65)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Tabla de estadísticas por segmento
    with st.expander("📋 Tabla de estadísticas"):
        stats = (
            df.groupby("segmento")[feat_cols]
            .agg(["mean", "std", "min", "max", "count"])
            .round(3)
        )
        st.dataframe(stats, use_container_width=True)


# ── Ejecutor de Pipeline Asíncrono ────────────────────────────────────────────

def render_pipeline_runner():
    if not st.session_state.get("pipeline_running"):
        return

    st.markdown("---")
    st.markdown("## 🚀 Ejecutando Pipeline")
    log_container = st.empty()
    progress      = st.progress(0, text="Iniciando…")

    def run_pipeline():
        """Hilo separado que ejecuta pipeline_runner.py como subproceso."""
        runner = os.path.join(str(ROOT), "pipeline_runner.py")
        proc   = subprocess.Popen(
            [sys.executable, runner],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        logs = []
        phase_keywords = {
            "FASE 1": 25, "FASE 2": 50, "FASE 3": 75, "FASE 4": 95
        }
        for line in proc.stdout:
            line = line.rstrip()
            logs.append(line)
            st.session_state["pipeline_log"] = logs.copy()
            for kw, pct in phase_keywords.items():
                if kw in line:
                    st.session_state["pipeline_pct"] = pct
        proc.wait()
        st.session_state["pipeline_running"] = False
        st.session_state["pipeline_pct"]     = 100

    if "pipeline_thread" not in st.session_state or \
            not st.session_state["pipeline_thread"].is_alive():
        t = threading.Thread(target=run_pipeline, daemon=True)
        t.start()
        st.session_state["pipeline_thread"] = t

    # Actualizar log en tiempo real
    logs = st.session_state.get("pipeline_log", [])
    pct  = st.session_state.get("pipeline_pct", 5)
    progress.progress(pct / 100, text=f"Progreso: {pct}%")
    log_container.text_area("Log del Pipeline", "\n".join(logs[-40:]),
                             height=300, label_visibility="collapsed")

    time.sleep(1)
    st.rerun()


# ── App Principal ─────────────────────────────────────────────────────────────

def main():
    st.title(f"🛒 {config.APP_TITLE}")
    st.markdown(
        "Pipeline de análisis híbrido **CPU (Dask + Multiprocessing) → GPU (CUDA)**  "
        "sobre datos de comportamiento de e-commerce (~9 GB)."
    )

    # Inicializar session_state
    if "pipeline_running" not in st.session_state:
        st.session_state["pipeline_running"] = False
    if "pipeline_log" not in st.session_state:
        st.session_state["pipeline_log"] = []
    if "pipeline_pct" not in st.session_state:
        st.session_state["pipeline_pct"] = 0

    # Cargar datos
    metrics     = load_metrics()
    features_df = load_user_features()
    gpu         = load_gpu_results()

    # Sidebar
    filters = render_sidebar(metrics, gpu)

    # Pipeline runner (si está activo)
    render_pipeline_runner()

    # Contenido principal
    if not metrics and not gpu:
        st.warning(
            "⚠️ No se encontraron resultados previos. "
            "Usa el botón **🚀 Ejecutar Pipeline Completo** en la barra lateral, "
            "o ejecuta `python pipeline_runner.py` en la terminal."
        )
        return

    if metrics:
        render_kpis(metrics)
        st.markdown("---")
        col_left, col_right = st.columns(2)
        with col_left:
            render_brand_revenue(metrics, filters)
        with col_right:
            render_hourly(metrics)
        st.markdown("---")
        render_monthly(metrics)
        st.markdown("---")

    render_segmentation(features_df, gpu, filters)
    st.markdown("---")
    render_similarity_heatmap(gpu)
    st.markdown("---")
    render_feature_filter(features_df, gpu, filters)

    # Footer
    st.markdown("---")
    gpu_mode = bool(int(gpu.get("gpu_mode", np.array([0]))[0])) if gpu else False
    st.caption(
        f"Dataset: [E-Commerce Behavior Data](https://www.kaggle.com/datasets/"
        f"mkechinov/ecommerce-behavior-data-from-multi-category-store) · "
        f"GPU: {'✅ CUDA activo' if gpu_mode else '⚠️ NumPy fallback'}"
    )


if __name__ == "__main__":
    main()
