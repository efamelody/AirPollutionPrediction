"""Malaysian API (Air Pollutant Index) conversion for PM2.5.

Uses US-EPA PM2.5 breakpoints adopted by Malaysia DOE for the API
(IPU) calculation - piecewise linear interpolation. This is the dominant
pollutant for haze. Values match DOE breakpoints.

Breakpoints (24h PM2.5 ug/m3 -> API):
"""

import numpy as np

# (c_low, c_high, i_low, i_high)
_PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

CATEGORIES = [
    (0, 50, "Good", "#00b050"),
    (51, 100, "Moderate", "#ffeb3b"),
    (101, 200, "Unhealthy", "#ff9800"),
    (201, 300, "Very Unhealthy", "#f44336"),
    (301, 500, "Hazardous", "#8b0000"),
]


def pm25_to_api(pm25: float) -> int:
    pm25 = float(np.clip(pm25, 0, 500.4))
    for c_low, c_high, i_low, i_high in _PM25_BREAKPOINTS:
        if pm25 <= c_high:
            if pm25 < c_low:
                return int(i_low)
            return int(round((i_high - i_low) / (c_high - c_low) * (pm25 - c_low) + i_low))
    return 500


def api_category(api: int) -> tuple[str, str]:
    for lo, hi, label, color in CATEGORIES:
        if lo <= api <= hi:
            return label, color
    return "Hazardous", "#8b0000"


def api_color_for_pm25(pm25: float) -> list[int]:
    label, hexcol = api_category(pm25_to_api(pm25))
    # hex to rgba
    hexcol = hexcol.lstrip("#")
    r, g, b = int(hexcol[0:2], 16), int(hexcol[2:4], 16), int(hexcol[4:6], 16)
    return [r, g, b, 180]
