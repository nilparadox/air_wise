from __future__ import annotations

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
REVERSE_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/reverse"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


def geocode_city(city: str) -> dict:
    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json",
    }
    r = requests.get(GEOCODE_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    results = data.get("results", [])
    if not results:
        raise RuntimeError(f"Could not find location for: {city}")

    item = results[0]
    return {
        "name": item.get("name", city),
        "admin1": item.get("admin1", ""),
        "country": item.get("country", ""),
        "latitude": item["latitude"],
        "longitude": item["longitude"],
    }


def fetch_open_meteo_bundle(city: str) -> dict:
    place = geocode_city(city)

    air_params = {
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "current": ",".join([
            "pm2_5",
            "pm10",
            "european_aqi",
            "us_aqi",
            "nitrogen_dioxide",
            "ozone",
            "sulphur_dioxide",
            "carbon_monoxide",
        ]),
        "timezone": "auto",
    }

    weather_params = {
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
        ]),
        "timezone": "auto",
    }

    air_res = requests.get(AIR_URL, params=air_params, timeout=20)
    air_res.raise_for_status()
    air_data = air_res.json()

    weather_res = requests.get(WEATHER_URL, params=weather_params, timeout=20)
    weather_res.raise_for_status()
    weather_data = weather_res.json()

    air_current = air_data.get("current", {})
    weather_current = weather_data.get("current", {})

    city_name = ", ".join([x for x in [place["name"], place["admin1"], place["country"]] if x])

    return {
        "city": city_name,
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "pm25": air_current.get("pm2_5"),
        "pm10": air_current.get("pm10"),
        "european_aqi": air_current.get("european_aqi"),
        "us_aqi": air_current.get("us_aqi"),
        "no2": air_current.get("nitrogen_dioxide"),
        "o3": air_current.get("ozone"),
        "so2": air_current.get("sulphur_dioxide"),
        "co": air_current.get("carbon_monoxide"),
        "temp_c": weather_current.get("temperature_2m"),
        "humidity": weather_current.get("relative_humidity_2m"),
        "wind_speed": weather_current.get("wind_speed_10m"),
    }
