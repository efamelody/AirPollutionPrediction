import numpy as np
import pandas as pd
import pydeck as pdk
from src.physics.api import api_category, pm25_to_api


def _haze_color(pm25: float) -> list[int]:
    api = pm25_to_api(pm25)
    label, hexcol = api_category(api)
    hexcol = hexcol.lstrip("#")
    r, g, b = int(hexcol[0:2], 16), int(hexcol[2:4], 16), int(hexcol[4:6], 16)
    return [r, g, b, 160]


def build_deck(
    hotspots_df: pd.DataFrame,
    receptors_df: pd.DataFrame,
    haze_grid: pd.DataFrame | None = None,
    meteo_grid: pd.DataFrame | None = None,
    show_haze: bool = True,
    show_wind: bool = True,
    show_rain: bool = False,
    show_receptors: bool = True,
) -> pdk.Deck:
    layers = []

    # 1. Haze heatmap (bottom layer)
    if show_haze and haze_grid is not None and not haze_grid.empty:
        hg = haze_grid.copy()
        hg["color"] = hg["pm25"].apply(_haze_color)
        # Heatmap for smooth appearance + column for precise cells
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                data=hg,
                get_position=["lon", "lat"],
                get_weight="pm25",
                radius_pixels=28,
                intensity=1.2,
                threshold=0.05,
                aggregation="MEAN",
                color_range=[[0, 176, 80], [255, 235, 59], [255, 152, 0], [244, 67, 54], [139, 0, 0], [80, 0, 0]],
            )
        )
        layers.append(
            pdk.Layer(
                "ColumnLayer",
                data=hg,
                get_position=["lon", "lat"],
                get_elevation="pm25 * 18",
                elevation_scale=1,
                radius=9000,
                get_fill_color="color",
                get_line_color=[40, 40, 40, 80],
                line_width_min_pixels=1,
                stroked=True,
                pickable=True,
                auto_highlight=True,
            )
        )

    # 2. Rain overlay (translucent circles where precip > threshold)
    if show_rain and meteo_grid is not None and not meteo_grid.empty:
        mg = meteo_grid.copy()
        mg["rain_alpha"] = (mg["precip"].clip(0, 20) / 20.0 * 160).astype(int)
        mg["rain_color"] = mg["rain_alpha"].apply(lambda a: [30, 144, 255, int(a)])
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=mg,
                get_position=["lon", "lat"],
                get_color="rain_color",
                get_radius="precip * 4000 + 8000",
                radius_min_pixels=6,
                radius_max_pixels=22,
                pickable=True,
            )
        )

    # 3. Wind vectors as line segments (from = upwind point, to = downwind offset)
    if show_wind and meteo_grid is not None and not meteo_grid.empty:
        mg = meteo_grid.copy()
        # Scale arrow length by wind speed (0.5-> small, 15-> long)
        scale = 22000  # meters per m/s
        mg["lon_to"] = mg["lon"] + (mg["u"] * scale / 111000 / np.cos(np.radians(mg["lat"])))
        mg["lat_to"] = mg["lat"] + (mg["v"] * scale / 111000)
        # Lines
        layers.append(
            pdk.Layer(
                "LineLayer",
                data=mg,
                get_source_position=["lon", "lat"],
                get_target_position=["lon_to", "lat_to"],
                get_color="[200, 200, 210, 220]",
                get_width=3,
                width_min_pixels=2,
            )
        )
        # Origin dots
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=mg,
                get_position=["lon", "lat"],
                get_color="[220, 220, 255, 200]",
                get_radius=4000,
                radius_min_pixels=3,
                pickable=True,
            )
        )

    # 4. Hotspots (orange, radius ∝ FRP)
    hs = hotspots_df.copy()
    if "posterior_prob" in hs.columns:
        hs["rgba"] = hs["posterior_prob"].apply(lambda p: [255, int(60 * (1 - float(p))), 0, 210])
    else:
        hs["rgba"] = [[255, 80, 0, 210]] * len(hs)
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=hs,
            get_position=["longitude", "latitude"],
            get_color="rgba",
            get_radius="frp * 320",
            radius_min_pixels=6,
            radius_max_pixels=44,
            stroked=True,
            get_line_color=[255, 255, 255, 200],
            line_width_min_pixels=1,
            pickable=True,
        )
    )

    # 5. Receptors (API-colored columns, height ∝ predicted or observed)
    if show_receptors and receptors_df is not None and not receptors_df.empty:
        rf = receptors_df.copy()
        # Use api_pred if present else pm25_obs, else pm25_pred
        val_col = None
        for c in ["pm25_pred", "pm25_obs", "api_pred"]:
            if c in rf.columns:
                val_col = c
                break
        if val_col is None:
            val_col = rf.columns[-1]
        # Color by API category of predicted value
        pm_col = "pm25_pred" if "pm25_pred" in rf.columns else ("pm25_obs" if "pm25_obs" in rf.columns else val_col)
        rf["receptor_color"] = rf[pm_col].apply(_haze_color) if pm_col in rf.columns else [[0, 128, 255, 180]] * len(rf)
        elev = "pm25_pred * 420" if "pm25_pred" in rf.columns else ("pm25_obs * 420" if "pm25_obs" in rf.columns else "api_pred * 120")
        layers.append(
            pdk.Layer(
                "ColumnLayer",
                data=rf,
                get_position=["longitude" if "longitude" in rf.columns else "lon", "latitude" if "latitude" in rf.columns else "lat"],
                get_elevation=elev,
                elevation_scale=1,
                radius=13500,
                get_fill_color="receptor_color",
                get_line_color=[255, 255, 255, 100],
                stroked=True,
                pickable=True,
                auto_highlight=True,
            )
        )

    view_state = pdk.ViewState(latitude=3.8, longitude=108.0, zoom=5.1, pitch=44, bearing=-4)

    tooltip = {
        "html": "<b>Haze</b> {pm25} µg/m³ API {api} {category}<br/><b>Hotspot</b> FRP {frp} posterior {posterior_prob}<br/><b>Wind</b> {wind_speed} m/s {wind_dir}° <b>Rain</b> {precip} mm/h",
        "style": {"backgroundColor": "#1a1a2e", "color": "white"},
    }

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/dark-v9",
    )
