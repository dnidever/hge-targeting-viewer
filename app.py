from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from astropy.table import Table
from scipy.spatial import cKDTree
from streamlit_plotly_events import plotly_events


st.set_page_config(page_title="HGE Targeting Photometry", layout="wide")

REQUIRED = ("l", "b", "jmag", "hmag", "kmag", "ak")


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


@st.cache_resource(show_spinner="Building spatial index…")
def make_spatial_index(catalog_key: str, _lon: np.ndarray, _lat: np.ndarray):
    """Build once per uploaded catalog; underscore args are not re-hashed on reruns."""
    lon_r = np.deg2rad(_lon)
    lat_r = np.deg2rad(_lat)
    xyz = np.column_stack(
        (np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r))
    )
    return cKDTree(xyz)


@st.cache_data(show_spinner=False)
def density_grid(catalog_key: str, bins: int, _lon: np.ndarray, _lat: np.ndarray):
    """Cache the expensive full-catalog histogram separately from the figure."""
    return np.histogram2d(_lon, _lat, bins=bins)


def unit_vector(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon, lat = np.deg2rad([lon_deg, lat_deg])
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def map_figure(
    data: pd.DataFrame, catalog_key: str, bins: int,
    center: tuple[float, float], radius: float,
) -> go.Figure:
    counts, l_edges, b_edges = density_grid(
        catalog_key, bins, data.l.to_numpy(), data.b.to_numpy()
    )
    l_centers = (l_edges[:-1] + l_edges[1:]) / 2
    b_centers = (b_edges[:-1] + b_edges[1:]) / 2
    fig = go.Figure(go.Heatmap(
        x=l_centers, y=b_centers, z=counts.T, colorscale="Viridis",
        showscale=False, hoverinfo="none",
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
        title="Stellar density — click to move the selection",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def density_scale_figure(
    data: pd.DataFrame, catalog_key: str, bins: int,
) -> go.Figure:
    """Render a horizontal density scale outside the interactive map figure."""
    counts, _, _ = density_grid(
        catalog_key, bins, data.l.to_numpy(), data.b.to_numpy()
    )
    scale_values = np.linspace(0.0, float(np.nanmax(counts)), 256)
    fig = go.Figure(go.Heatmap(
        x=scale_values, y=[0], z=scale_values[np.newaxis, :],
        colorscale="Viridis", showscale=False, hoverinfo="skip",
    ))
    fig.update_layout(
        height=90, margin=dict(l=10, r=10, t=0, b=35),
        xaxis_title="Stars per density bin",
        yaxis={"visible": False, "fixedrange": True},
        xaxis={"fixedrange": True},
    )
    return fig


def density_bin_count(
    data: pd.DataFrame, catalog_key: str, bins: int,
    center: tuple[float, float],
) -> int:
    counts, l_edges, b_edges = density_grid(
        catalog_key, bins, data.l.to_numpy(), data.b.to_numpy()
    )
    i = np.clip(np.searchsorted(l_edges, center[0], side="right") - 1, 0, counts.shape[0] - 1)
    j = np.clip(np.searchsorted(b_edges, center[1], side="right") - 1, 0, counts.shape[1] - 1)
    return int(counts[i, j])


def cmd_figure(
    selected: pd.DataFrame, center: tuple[float, float], radius: float,
    max_points: int, auto_range: bool,
    color_min: float, color_max: float, h_bright: float, h_faint: float,
) -> go.Figure:
    total = len(selected)
    if total > max_points:
        displayed = selected.sample(max_points, random_state=42)
    else:
        displayed = selected
    color = displayed.jmag - displayed.kmag
    custom = (
        np.column_stack((displayed.l, displayed.b, displayed.jmag, displayed.kmag, displayed.ak))
        if len(displayed) else None
    )
    # Use the SVG scatter renderer rather than Scattergl so the CMD works in
    # browsers or remote-desktop sessions without WebGL support.
    fig = go.Figure(go.Scatter(
        x=color, y=displayed.hmag, mode="markers",
        marker={
            "size": 3, "opacity": 0.55, "color": displayed.ak,
            "colorscale": "Viridis", "showscale": True,
            "colorbar": {"title": "A(Ks)"},
        },
        customdata=custom,
        hovertemplate="J−Ks=%{x:.3f}<br>H=%{y:.3f}<br>A(Ks)=%{customdata[4]:.3f}<br>l=%{customdata[0]:.4f}°<br>b=%{customdata[1]:.4f}°<extra></extra>",
    ))
    if auto_range:
        fig.update_yaxes(autorange="reversed")
    else:
        fig.update_xaxes(range=[color_min, color_max])
        fig.update_yaxes(range=[h_faint, h_bright], autorange=False)
    fig.update_layout(
        title=f"CMD: {total:,} stars within {radius:.3f}° of ({center[0]:.4f}°, {center[1]:.4f}°)",
        xaxis_title="J − Ks (mag)", yaxis_title="H (mag)", height=620,
        margin=dict(l=10, r=10, t=60, b=10), uirevision="cmd",
    )
    if total == 0:
        fig.add_annotation(
            text="No stars in this region", x=0.5, y=0.5,
            xref="paper", yref="paper", showarrow=False, font={"size": 18},
        )
    elif total > max_points:
        fig.add_annotation(
            text=f"Showing a random {max_points:,} of {total:,} stars",
            x=0.99, y=0.01, xref="paper", yref="paper",
            xanchor="right", yanchor="bottom", showarrow=False,
        )
    return fig


st.title("HGE Targeting Photometry Viewer")
uploaded = st.sidebar.file_uploader("Catalog", type=["fits", "fit", "fz", "csv", "parquet", "pq", "txt", "dat"])
st.sidebar.caption("Required columns: l, b, jmag, hmag, kmag, ak")

if uploaded is None:
    st.info("Upload a FITS, CSV, Parquet, or whitespace-delimited catalog to begin.")
    st.stop()

try:
    file_bytes = uploaded.getvalue()
    catalog = read_catalog(file_bytes, uploaded.name)
except Exception as exc:
    st.error(f"Could not read the catalog: {exc}")
    st.stop()

radius = st.sidebar.number_input("Circular region radius (deg)", 0.001, 5.0, 0.06, 0.01, format="%.3f")
bins = st.sidebar.slider("Density-map bins per axis", 30, 250, 120, 10)
max_cmd_points = st.sidebar.select_slider(
    "Maximum CMD points rendered",
    options=[500, 1_000, 2_000, 5_000, 10_000],
    value=2_000,
    help="The CMD uses a non-WebGL renderer. Lower values are faster for large selections.",
)
auto_cmd_range = st.sidebar.toggle(
    "Automatic CMD axis ranges", value=False,
    help="When enabled, the axes adjust to the stars in each newly selected region.",
)
with st.sidebar.expander("Fixed CMD axis ranges", expanded=True):
    color_min = st.number_input("Minimum J − Ks", value=-0.5, step=0.1, format="%.2f")
    color_max = st.number_input("Maximum J − Ks", value=5.0, step=0.1, format="%.2f")
    h_bright = st.number_input("Bright H limit", value=7.0, step=0.5, format="%.2f")
    h_faint = st.number_input("Faint H limit", value=18.0, step=0.5, format="%.2f")
if not auto_cmd_range and (color_min >= color_max or h_bright >= h_faint):
    st.sidebar.error("CMD minima must be smaller than their corresponding maxima.")
    st.stop()
catalog_key = f"{uploaded.name}:{len(file_bytes)}:{len(catalog)}"
initial_l = float(np.nanmedian(catalog.l))
initial_b = float(np.nanmedian(catalog.b))
if "map_center" not in st.session_state:
    st.session_state.map_center = (initial_l, initial_b)

with st.sidebar.expander("Set center manually"):
    manual_l = st.number_input("l (deg)", value=float(st.session_state.map_center[0]), format="%.5f")
    manual_b = st.number_input("b (deg)", value=float(st.session_state.map_center[1]), format="%.5f")
    if st.button("Use manual center"):
        st.session_state.map_center = (manual_l, manual_b)

@st.fragment
def interactive_viewer():
    """Rerun only the map/CMD panel, rather than the full app, after a click."""
    left, right = st.columns([1.15, 1.0])
    with left:
        st.caption("Click a density bin to update the CMD.")
        event = plotly_events(
            map_figure(catalog, catalog_key, bins, st.session_state.map_center, radius),
            hover_event=False, click_event=True, select_event=False,
            override_height=620, key="density_map",
        )
        st.plotly_chart(
            density_scale_figure(catalog, catalog_key, bins),
            use_container_width=True, config={"displayModeBar": False},
        )
        map_status = st.empty()

    if event:
        point = event[-1]
        if "x" in point and "y" in point:
            new_center = (float(point["x"]), float(point["y"]))
            if new_center != st.session_state.map_center:
                st.session_state.map_center = new_center
                st.rerun(scope="fragment")

    bin_count = density_bin_count(catalog, catalog_key, bins, st.session_state.map_center)
    map_status.caption(
        f"Selected center: **l={st.session_state.map_center[0]:.4f}°, "
        f"b={st.session_state.map_center[1]:.4f}°** · Density bin: **{bin_count:,} stars**"
    )

    tree = make_spatial_index(
        catalog_key, catalog.l.to_numpy(copy=False), catalog.b.to_numpy(copy=False)
    )
    chord_radius = 2 * np.sin(np.deg2rad(radius) / 2)
    indices = tree.query_ball_point(unit_vector(*st.session_state.map_center), chord_radius)
    selected = catalog.iloc[indices]

    with right:
        st.caption(f"Selected region: **{len(selected):,} stars**")
        st.plotly_chart(
            cmd_figure(
                selected, st.session_state.map_center, radius, max_cmd_points,
                auto_cmd_range, color_min, color_max, h_bright, h_faint,
            ),
            use_container_width=True,
        )


interactive_viewer()

st.caption(f"Catalog: {len(catalog):,} valid stars. The circular query uses exact angular separation on the sphere.")
