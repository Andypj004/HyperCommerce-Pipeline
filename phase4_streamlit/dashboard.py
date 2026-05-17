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

SEG_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
PIPELINE_LOG_FILE = os.path.join(config.OUTPUT_DIR, "pipeline_run.log")
PIPELINE_PID_FILE = os.path.join(config.OUTPUT_DIR, "pipeline.pid")


# ── Helpers de pipeline ───────────────────────────────────────────────────────

def _pipeline_is_running() -> bool:
    if not os.path.exists(PIPELINE_PID_FILE):
        return False
    try:
        pid = int(open(PIPELINE_PID_FILE).read().strip())
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError, OSError):
        return False

def _read_log_tail(n: int = 80) -> str:
    if not os.path.exists(PIPELINE_LOG_FILE):
        return "(sin log aún — el pipeline no ha sido lanzado)"
    try:
        lines = open(PIPELINE_LOG_FILE, "r", errors="replace").readlines()
        return "".join(lines[-n:]) if lines else "(log vacío)"
    except Exception as e:
        return f"(error leyendo log: {e})"

def _log_pct() -> int:
    log = _read_log_tail(400)
    if "PIPELINE COMPLETADO" in log: return 100
    if "FASE 3 completada"   in log: return 90
    if "FASE 3"              in log: return 65
    if "FASE 2 completada"   in log: return 60
    if "FASE 2"              in log: return 35
    if "FASE 1 completada"   in log: return 30
    if "FASE 1"              in log: return 10
    return 3

def _start_pipeline():
    open(PIPELINE_LOG_FILE, "w").close()
    proc = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "pipeline_runner.py")],
        stdout=open(PIPELINE_LOG_FILE, "w"),
        stderr=subprocess.STDOUT,
        close_fds=True,
    )
    open(PIPELINE_PID_FILE, "w").write(str(proc.pid))


# ── Loaders sin caché ─────────────────────────────────────────────────────────

def load_metrics() -> dict:
    if not os.path.exists(config.METRICS_FILE):
        return {}
    try:
        return json.load(open(config.METRICS_FILE))
    except Exception:
        return {}

def load_user_features() -> pd.DataFrame:
    if not os.path.exists(config.CPU_RESULT):
        return pd.DataFrame()
    try:
        return pd.read_parquet(config.CPU_RESULT)
    except Exception:
        return pd.DataFrame()

def load_gpu_results() -> dict:
    for path in [config.GPU_RESULT + ".npz", config.GPU_RESULT]:
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=True)
                return {k: data[k] for k in data.files}
            except Exception:
                pass
    return {}


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(metrics: dict, gpu: dict) -> dict:
    st.sidebar.title("⚙️ Control del Pipeline")
    running = _pipeline_is_running()

    st.sidebar.markdown("### Estado de Fases")
    for label, exists in [
        ("Fase 1 — Dask",      os.path.exists(config.PARQUET_DIR)),
        ("Fase 2 — Multiproc", os.path.exists(config.CPU_RESULT)),
        ("Fase 3 — GPU/NumPy", any(os.path.exists(p) for p in
                                    [config.GPU_RESULT, config.GPU_RESULT + ".npz"])),
    ]:
        icon = "✅" if exists else ("⏳" if running else "❌")
        st.sidebar.markdown(f"{icon} {label}")

    st.sidebar.markdown("---")
    if running:
        st.sidebar.warning("⏳ Pipeline ejecutándose…")
    else:
        if st.sidebar.button("🚀 Ejecutar Pipeline",
                             use_container_width=True, type="primary"):
            _start_pipeline()
            st.session_state["pipeline_launched"] = True
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filtros")
    filters = {}
    if metrics:
        filters["price_bucket"] = st.sidebar.selectbox(
            "Bucket de Precio", ["Todos", "low", "mid", "high"])
        top_brands = ["Todas"] + list(metrics.get("revenue_by_brand", {}).keys())[:20]
        filters["brand"] = st.sidebar.selectbox("Marca", top_brands)
    if gpu:
        n_seg = int(gpu.get("seg_count", np.zeros(5)).shape[0])
        filters["segment"] = st.sidebar.selectbox(
            "Segmento GPU", ["Todos"] + [f"Segmento {i}" for i in range(n_seg)])
    return filters


# ── Panel de log ──────────────────────────────────────────────────────────────

def render_log_panel() -> bool:
    """
    Dibuja el panel de log y barra de progreso.
    Devuelve True si el pipeline está corriendo (para que main haga el rerun al final).
    """
    running   = _pipeline_is_running()
    launched  = st.session_state.get("pipeline_launched", False)

    # Mostrar panel solo si fue lanzado alguna vez
    if not launched and not os.path.exists(PIPELINE_LOG_FILE):
        return False

    pct = _log_pct()

    st.markdown("---")
    st.markdown("## 🚀 Ejecución del Pipeline")

    col_prog, col_status = st.columns([4, 1])
    with col_prog:
        if running:
            st.progress(pct / 100,
                        text=f"⏳ Progreso estimado: {pct}%  —  actualizando cada 2s…")
        else:
            st.progress(1.0, text="✅ Pipeline completado")

    with col_status:
        if running:
            st.markdown("🟢 **En curso**")
        else:
            st.markdown("✔️ **Finalizado**")

    # Log en tiempo real
    log_content = _read_log_tail(60)
    st.code(log_content, language="bash")

    # Nota de ayuda mientras corre
    if running:
        st.caption("📋 Las líneas más recientes aparecen abajo del recuadro. "
                   "La página se refresca sola cada 2 segundos.")

    return running


# ── Visualizaciones ───────────────────────────────────────────────────────────

def render_kpis(metrics: dict):
    st.markdown("## 📊 KPIs Globales")
    rev   = metrics.get("total_revenue", 0)
    users = metrics.get("unique_users", 0)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Eventos",    f"{metrics.get('total_events', 0):,}")
    c2.metric("Revenue Total",    f"${rev:,.0f}")
    c3.metric("Usuarios Únicos",  f"{users:,}")
    c4.metric("Productos Únicos", f"{metrics.get('unique_products', 0):,}")
    c5.metric("Ticket Promedio",  f"${rev/users:,.2f}" if users else "$0")

def render_brand_revenue(metrics: dict):
    brand_rev = metrics.get("revenue_by_brand", {})
    if not brand_rev:
        return
    df = (pd.DataFrame({"brand": list(brand_rev.keys()),
                         "revenue": list(brand_rev.values())})
            .sort_values("revenue", ascending=True).tail(30))
    fig = px.bar(df, x="revenue", y="brand", orientation="h",
                 color="revenue", color_continuous_scale="Viridis",
                 title="Revenue por Marca (Top 30)",
                 labels={"revenue": "Revenue (USD)", "brand": "Marca"})
    fig.update_layout(height=500, coloraxis_showscale=False,
                      margin=dict(l=120, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

def render_hourly(metrics: dict):
    hourly = metrics.get("hourly_stats", [])
    if not hourly:
        return
    df  = pd.DataFrame(hourly)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["hour"], y=df["count"], name="Transacciones",
                         marker_color="#636EFA", opacity=0.6), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["hour"], y=df["mean"], name="Ticket Medio (USD)",
                             mode="lines+markers",
                             line=dict(color="#EF553B", width=2)), secondary_y=True)
    fig.update_layout(title="Distribución Horaria de Compras",
                      xaxis_title="Hora (UTC)", height=350,
                      legend=dict(x=0.01, y=0.99))
    fig.update_yaxes(title_text="Transacciones",   secondary_y=False)
    fig.update_yaxes(title_text="Ticket Promedio", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

def render_monthly(metrics: dict):
    monthly = metrics.get("monthly_stats", [])
    if not monthly:
        return
    df = pd.DataFrame(monthly)
    df["periodo"] = (df["year"].astype(str) + "-"
                     + df["month"].astype(str).str.zfill(2))
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(df, x="periodo", y="sum", markers=True,
                      title="Revenue Mensual Total",
                      labels={"sum": "Revenue (USD)", "periodo": "Período"})
        fig.update_traces(line_color="#00CC96", line_width=2)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(df, x="periodo", y="count", title="Transacciones por Mes",
                     color="count", color_continuous_scale="Blues")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

def render_segmentation(features_df: pd.DataFrame, gpu: dict, filters: dict):
    st.markdown("## 🤖 Segmentación de Usuarios (GPU / CUDA)")
    if features_df.empty or not gpu:
        st.info("Sin resultados de Fase 2/3.")
        return
    seg_ids = gpu.get("seg_ids", np.array([]))
    if len(seg_ids) == 0:
        return
    FEATS     = ["frequency","total_spend","avg_price","std_price",
                 "entropy_cat","brand_loyalty","weekend_ratio","peak_hour"]
    feat_cols = [c for c in FEATS if c in features_df.columns]
    n         = min(len(seg_ids), len(features_df))
    sub_df    = features_df[feat_cols].iloc[:n].fillna(0)
    pca       = PCA(n_components=2)
    coords    = pca.fit_transform(sub_df.values)
    plot_df   = pd.DataFrame({
        "PC1": coords[:, 0], "PC2": coords[:, 1],
        "Segmento":    [f"Seg {s}" for s in seg_ids[:n]],
        "frequency":   sub_df["frequency"].values,
        "total_spend": sub_df["total_spend"].values,
    })
    seg_filter = filters.get("segment", "Todos")
    if seg_filter != "Todos":
        plot_df = plot_df[plot_df["Segmento"] == f"Seg {seg_filter.split()[-1]}"]
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="Segmento",
                         size="frequency", hover_data=["total_spend"],
                         color_discrete_sequence=SEG_COLORS,
                         title="Mapa de Segmentos de Usuarios (PCA 2D)",
                         labels={
                             "PC1": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                             "PC2": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                         })
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        seg_mean   = gpu.get("seg_mean")
        feat_names = list(gpu.get("feature_names", np.array(FEATS)))
        if seg_mean is not None:
            fig_r = go.Figure()
            for s in range(seg_mean.shape[0]):
                vals = list(seg_mean[s]) + [seg_mean[s][0]]
                cats = feat_names + [feat_names[0]]
                fig_r.add_trace(go.Scatterpolar(
                    r=vals, theta=cats, fill="toself",
                    name=f"Seg {s}", line_color=SEG_COLORS[s % 5], opacity=0.7))
            fig_r.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title="Perfil de Segmentos (Normalizado)", height=400)
            st.plotly_chart(fig_r, use_container_width=True)

def render_similarity_heatmap(gpu: dict):
    st.markdown("## 🔥 Matriz de Similitud Coseno (GPU)")
    sim = gpu.get("sim_matrix")
    if sim is None:
        st.info("Sin resultados de Fase 3.")
        return
    n   = min(100, sim.shape[0])
    fig = px.imshow(sim[:n, :n], color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title=f"Similitud Coseno — primeros {n}×{n} usuarios",
                    labels={"color": "Similitud"})
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    gpu_mode = bool(int(gpu.get("gpu_mode", np.array([0]))[0]))
    st.caption(f"⚡ Modo: {'**GPU CUDA**' if gpu_mode else '**CPU NumPy fallback**'} · "
               f"Dimensión completa: {sim.shape[0]}×{sim.shape[0]}")

def render_feature_filter(features_df: pd.DataFrame, gpu: dict, filters: dict):
    st.markdown("## 🔍 Análisis de Features por Segmento")
    if features_df.empty or not gpu:
        return
    seg_ids = gpu.get("seg_ids", np.array([]))
    if len(seg_ids) == 0:
        return
    FEATS     = ["frequency","total_spend","avg_price","std_price",
                 "entropy_cat","brand_loyalty","weekend_ratio","peak_hour"]
    n         = min(len(seg_ids), len(features_df))
    df        = features_df.iloc[:n].copy()
    df["segmento"] = [f"Seg {s}" for s in seg_ids[:n]]
    seg_filter = filters.get("segment", "Todos")
    if seg_filter != "Todos":
        df = df[df["segmento"] == f"Seg {seg_filter.split()[-1]}"]
    feat_cols     = [c for c in FEATS if c in df.columns]
    selected_feat = st.selectbox("Feature a visualizar", feat_cols, index=1)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(df, x="segmento", y=selected_feat, color="segmento",
                     color_discrete_sequence=SEG_COLORS,
                     title=f"Distribución de '{selected_feat}'", points=False)
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df, x=selected_feat, color="segmento",
                           color_discrete_sequence=SEG_COLORS,
                           nbins=40, barmode="overlay", opacity=0.65,
                           title=f"Histograma de '{selected_feat}'")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 Tabla de estadísticas por segmento"):
        stats = (df.groupby("segmento")[feat_cols]
                   .agg(["mean","std","min","max","count"])
                   .round(3))
        st.dataframe(stats, use_container_width=True)


# ── App Principal ─────────────────────────────────────────────────────────────

def main():
    if "pipeline_launched" not in st.session_state:
        st.session_state["pipeline_launched"] = False

    st.title(f"🛒 {config.APP_TITLE}")
    st.caption("Pipeline híbrido: Dask → Multiprocessing → GPU (CUDA) → Dashboard")

    # Cargar datos del disco
    metrics     = load_metrics()
    features_df = load_user_features()
    gpu         = load_gpu_results()

    # Sidebar
    filters = render_sidebar(metrics, gpu)

    # ── Panel de log: se dibuja PRIMERO, devuelve si sigue corriendo ──────────
    pipeline_running = render_log_panel()

    # ── Dashboard de resultados ───────────────────────────────────────────────
    if not metrics and not gpu:
        if not pipeline_running:
            st.info("⚠️ Sin resultados aún. Presiona **🚀 Ejecutar Pipeline** "
                    "en la barra lateral.")
    else:
        if metrics:
            st.markdown("---")
            render_kpis(metrics)
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1: render_brand_revenue(metrics)
            with c2: render_hourly(metrics)
            st.markdown("---")
            render_monthly(metrics)
            st.markdown("---")

        render_segmentation(features_df, gpu, filters)
        st.markdown("---")
        render_similarity_heatmap(gpu)
        st.markdown("---")
        render_feature_filter(features_df, gpu, filters)

        st.markdown("---")
        gpu_mode = bool(int(gpu.get("gpu_mode", np.array([0]))[0])) if gpu else False
        st.caption(
            f"Dataset: [Kaggle E-Commerce Behavior](https://www.kaggle.com/datasets/"
            f"mkechinov/ecommerce-behavior-data-from-multi-category-store) · "
            f"GPU: {'✅ CUDA activo' if gpu_mode else '⚠️ NumPy fallback'}"
        )

    # ── Auto-refresh AL FINAL, después de renderizar todo ────────────────────
    # Así Streamlit muestra el log y las gráficas actuales ANTES de dormir.
    # Al despertar hace rerun → el script corre de nuevo desde arriba con datos frescos.
    if pipeline_running:
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()