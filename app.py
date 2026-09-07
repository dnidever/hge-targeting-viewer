from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from astropy.table import Table
from scipy.spatial import cKDTree
from streamlit_plotly_events import plotly_events


st.set_page_config(page_title="HGE Targeting Photometry", layout="wide")

REQUIRED = ("l", "b", "jmag", "hmag", "kmag")


@st.cache_data(show_spinner="Reading catalog…")
def read_catalog(file_bytes: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    from io import BytesIO

    buffer = BytesIO(file_bytes)
    if suffix == ".csv":
        frame = pd.read_csv(buffer)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(buffer)
    elif suffix in {".fits", ".fit", ".fz"}:
        frame = Table.read(buffer, format="fits").to_pandas()
    else:
        try:
            frame = Table.read(buffer).to_pandas()
        except Exception:
            buffer.seek(0)
            frame = pd.read_csv(buffer, delim_whitespace=True, comment="#")

    frame.columns = [str(name).strip().lower() for name in frame.columns]
    missing = [name for name in REQUIRED if name not in frame]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    for name in REQUIRED:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=list(REQUIRED)).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def make_spatial_index(lon: tuple[float, ...], lat: tuple[float, ...]):
    lon_r = np.deg2rad(np.asarray(lon))
    lat_r = np.deg2rad(np.asarray(lat))
    xyz = np.column_stack(
        (np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r))
    )
    return cKDTree(xyz)


def unit_vector(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon, lat = np.deg2rad([lon_deg, lat_deg])
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def map_figure(data: pd.DataFrame, bins: int, center: tuple[float, float], radius: float) -> go.Figure:
    counts, l_edges, b_edges = np.histogram2d(data.l, data.b, bins=bins)
    l_centers = (l_edges[:-1] + l_edges[1:]) / 2
    b_centers = (b_edges[:-1] + b_edges[1:]) / 2
    fig = go.Figure(go.Heatmap(
        x=l_centers, y=b_centers, z=counts.T, colorscale="Viridis",
        colorbar={"title": "Stars/bin"}, hovertemplate="l=%{x:.4f}°<br>b=%{y:.4f}°<br>%{z:,.0f} stars<extra></extra>",
    ))
    theta = np.linspace(0, 2 * np.pi, 181)
    cos_b = max(np.cos(np.deg2rad(center[1])), 0.05)
    fig.add_trace(go.Scatter(
        x=center[0] + radius * np.cos(theta) / cos_b,
        y=center[1] + radius * np.sin(theta), mode="lines",
        line={"color": "white", "width": 2}, hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        xaxis_title="Galactic longitude l (deg)", yaxis_title="Galactic latitude b (deg)",
        margin=dict(l=10, r=10, t=35, b=10), height=620, uirevision="density-map",
        title="Stellar density — hover to move the selection",
    )
    return fig


def cmd_figure(selected: pd.DataFrame, center: tuple[float, float], radius: float) -> go.Figure:
    color = selected.jmag - selected.kmag
    custom = np.column_stack((selected.l, selected.b, selected.jmag, selected.kmag)) if len(selected) else None
    fig = go.Figure(go.Scattergl(
        x=color, y=selected.hmag, mode="markers",
        marker={"size": 4, "opacity": 0.45, "color": selected.hmag, "colorscale": "Plasma", "showscale": False},
        customdata=custom,
        hovertemplate="J−Ks=%{x:.3f}<br>H=%{y:.3f}<br>l=%{customdata[0]:.4f}°<br>b=%{customdata[1]:.4f}°<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=f"CMD: {len(selected):,} stars within {radius:.3f}° of ({center[0]:.4f}°, {center[1]:.4f}°)",
        xaxis_title="J − Ks (mag)", yaxis_title="H (mag)", height=620,
        margin=dict(l=10, r=10, t=60, b=10), uirevision="cmd",
    )
    return fig


st.title("HGE Targeting Photometry Viewer")
uploaded = st.sidebar.file_uploader("Catalog", type=["fits", "fit", "fz", "csv", "parquet", "pq", "txt", "dat"])
st.sidebar.caption("Required columns: l, b, jmag, hmag, kmag")

if uploaded is None:
    st.info("Upload a FITS, CSV, Parquet, or whitespace-delimited catalog to begin.")
    st.stop()

try:
    catalog = read_catalog(uploaded.getvalue(), uploaded.name)
except Exception as exc:
    st.error(f"Could not read the catalog: {exc}")
    st.stop()

radius = st.sidebar.number_input("Circular region radius (deg)", 0.001, 5.0, 0.10, 0.01, format="%.3f")
bins = st.sidebar.slider("Density-map bins per axis", 30, 250, 120, 10)
initial_l = float(np.nanmedian(catalog.l))
initial_b = float(np.nanmedian(catalog.b))
if "map_center" not in st.session_state:
    st.session_state.map_center = (initial_l, initial_b)

with st.sidebar.expander("Set center manually"):
    manual_l = st.number_input("l (deg)", value=float(st.session_state.map_center[0]), format="%.5f")
    manual_b = st.number_input("b (deg)", value=float(st.session_state.map_center[1]), format="%.5f")
    if st.button("Use manual center"):
        st.session_state.map_center = (manual_l, manual_b)

left, right = st.columns([1.15, 1.0])
with left:
    event = plotly_events(
        map_figure(catalog, bins, st.session_state.map_center, radius),
        hover_event=True, click_event=True, select_event=False,
        override_height=620, key="density_map",
    )

if event:
    point = event[-1]
    if "x" in point and "y" in point:
        new_center = (float(point["x"]), float(point["y"]))
        if new_center != st.session_state.map_center:
            st.session_state.map_center = new_center

tree = make_spatial_index(tuple(catalog.l.to_numpy()), tuple(catalog.b.to_numpy()))
chord_radius = 2 * np.sin(np.deg2rad(radius) / 2)
indices = tree.query_ball_point(unit_vector(*st.session_state.map_center), chord_radius)
selected = catalog.iloc[indices]

with right:
    st.plotly_chart(cmd_figure(selected, st.session_state.map_center, radius), use_container_width=True)

st.caption(f"Catalog: {len(catalog):,} valid stars. The circular query uses exact angular separation on the sphere.")
