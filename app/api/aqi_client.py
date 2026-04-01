from __future__ import annotations

import time
import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

HEADERS = {
    "User-Agent": "ZetaQ-AirWise/1.0"
}

# very small in-memory cache for repeated city searches
_CACHE = {}
CACHE_TTL_SECONDS = 900  # 15 minutes


def _cache_get(key: str):
    item = _CACHE.get(key)
    if not item:
        return None
    ts, value = item
    if time.time() - ts > CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return value


def _cache_set(key: str, value):
    _CACHE[key] = (time.time(), value)


def geocode_city(city: str) -> dict:
    cache_key = f"geo::{city.strip().lower()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json",
    }

    r = requests.get(GEOCODE_URL, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()

    results = data.get("results", [])
    if not results:
        raise RuntimeError(f"Could not find location for: {city}")

    item = results[0]
    result = {
        "name": item.get("name", city),
        "admin1": item.get("admin1", ""),
        "country": item.get("country", ""),
        "latitude": item["latitude"],
        "longitude": item["longitude"],
    }
    _cache_set(cache_key, result)
    return result


def _fetch_air(latitude: float, longitude: float) -> dict:
    params = {
        "latitude": latitude,
        "longitude": longitude,
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

    r = requests.get(AIR_URL, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()


def _fetch_weather(latitude: float, longitude: float) -> dict:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
        ]),
        "timezone": "auto",
    }

    r = requests.get(WEATHER_URL, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()


def fetch_open_meteo_bundle(city: str) -> dict:
    cache_key = f"bundle::{city.strip().lower()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    place = geocode_city(city)

    air_data = _fetch_air(place["latitude"], place["longitude"])
    air_current = air_data.get("current", {})

    # Weather is optional. If it fails, continue with defaults.
    temp_c = 30.0
    humidity = 60.0
    wind_speed = 5.0
    weather_note = None

    try:
        weather_data = _fetch_weather(place["latitude"], place["longitude"])
        weather_current = weather_data.get("current", {})
        temp_c = weather_current.get("temperature_2m", temp_c)
        humidity = weather_current.get("relative_humidity_2m", humidity)
        wind_speed = weather_current.get("wind_speed_10m", wind_speed)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            weather_note = "Weather API temporarily rate-limited; using fallback weather values."
        else:
            weather_note = "Weather data unavailable; using fallback weather values."
    except Exception:
        weather_note = "Weather data unavailable; using fallback weather values."

    city_name = ", ".join([x for x in [place["name"], place["admin1"], place["country"]] if x])

    result = {
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
        "temp_c": temp_c,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "weather_note": weather_note,
    }

    _cache_set(cache_key, result)
    return result
