"""Weather via Open-Meteo — 100% free, no API key, no signup."""

import httpx

from core.config import (
    WEATHER_DEFAULT_CITY,
    WEATHER_DEFAULT_LATITUDE,
    WEATHER_DEFAULT_LONGITUDE,
)
from core.logger import logger

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT_SECONDS = 5.0

# WMO Weather interpretation codes → human-readable text
_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _weather_code_to_text(code):
    try:
        return _WEATHER_CODES.get(int(code), "Unknown")
    except (TypeError, ValueError):
        return "Unknown"


def get_weather(lat=None, lon=None, city=None):
    """Fetch current weather from Open-Meteo.

    Returns a formatted string on success, empty string on failure.
    """
    lat = float(lat or WEATHER_DEFAULT_LATITUDE)
    lon = float(lon or WEATHER_DEFAULT_LONGITUDE)
    city = str(city or WEATHER_DEFAULT_CITY or "Cairo").strip()

    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "timezone": "auto",
        }
        r = httpx.get(_OPEN_METEO_URL, params=params, timeout=_TIMEOUT_SECONDS)
        if r.status_code != 200:
            logger.warning("Open-Meteo returned status %s", r.status_code)
            return ""

        data = r.json().get("current", {})
        temp = data.get("temperature_2m", "?")
        humidity = data.get("relative_humidity_2m", "?")
        wind = data.get("wind_speed_10m", "?")
        code = data.get("weather_code", 0)
        condition = _weather_code_to_text(code)

        return (
            f"Weather in {city}: {condition}, {temp}°C, "
            f"humidity {humidity}%, wind {wind} km/h"
        )
    except Exception as exc:
        logger.warning("Weather fetch failed: %s", exc)
        return ""
