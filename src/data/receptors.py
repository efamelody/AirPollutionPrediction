import pandas as pd


def fetch_malaysia_receptors() -> pd.DataFrame:
    stations = {
        "station_name": ["Kuala Lumpur", "Johor Bahru", "Kuching", "Ipoh", "Kota Bharu"],
        "latitude": [3.1390, 1.4927, 1.5533, 4.5975, 6.1254],
        "longitude": [101.6869, 103.7414, 110.3592, 101.0901, 102.2381],
        "pm25_obs": [85.4, 62.1, 112.0, 45.8, 38.2],
    }
    return pd.DataFrame(stations)
