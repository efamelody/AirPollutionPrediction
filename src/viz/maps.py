import pydeck as pdk
import pandas as pd


def build_deck(hotspots_df: pd.DataFrame, receptors_df: pd.DataFrame) -> pdk.Deck:
    # Precompute RGBA for hotspots (posterior-weighted red)
    hs = hotspots_df.copy()
    if "posterior_prob" in hs.columns:
        # Use posterior to modulate green channel; alpha encodes confidence
        hs["rgba"] = hs["posterior_prob"].apply(lambda p: [255, int(60 * (1 - float(p))), 0, 200])
    else:
        hs["rgba"] = [[255, 60, 0, 200]] * len(hs)

    layer_hotspots = pdk.Layer(
        "ScatterplotLayer",
        data=hs,
        get_position=["longitude", "latitude"],
        get_color="rgba",
        get_radius="frp * 250",
        radius_min_pixels=5,
        radius_max_pixels=40,
        pickable=True,
    )

    layer_receptors = pdk.Layer(
        "ColumnLayer",
        data=receptors_df,
        get_position=["longitude", "latitude"],
        get_elevation="pm25_obs * 500",
        elevation_scale=1,
        radius=15000,
        get_fill_color="[0, 128, 255, 180]",
        pickable=True,
    )

    view_state = pdk.ViewState(latitude=2.5, longitude=105.0, zoom=5, pitch=45)

    tooltip = {"html": "<b>Lat:</b> {latitude}<br/><b>Lon:</b> {longitude}<br/><b>FRP:</b> {frp}<br/><b>Posterior:</b> {posterior_prob}"}

    return pdk.Deck(layers=[layer_hotspots, layer_receptors], initial_view_state=view_state, tooltip=tooltip)
