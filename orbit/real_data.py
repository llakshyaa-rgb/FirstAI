"""Live, source-attributed public-data retrieval for Orbit.

Only fixed, allow-listed public API hosts are queried. User text is used to
choose a dataset and search terms, never as an arbitrary URL or shell command.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone

TIMEOUT = 18
MAX_BYTES = 3 * 1024 * 1024
HEADERS = {
    "User-Agent": "FirstAI-Orbit/1.0 (public-data retrieval)",
    "Accept": "application/json",
}
CURRENCY_NAMES = {
    "inr": "INR", "rupee": "INR", "rupees": "INR", "usd": "USD",
    "dollar": "USD", "dollars": "USD", "eur": "EUR", "euro": "EUR",
    "gbp": "GBP", "pound": "GBP", "jpy": "JPY", "yen": "JPY",
    "aud": "AUD", "cad": "CAD", "chf": "CHF", "sgd": "SGD",
}
INDICATORS = {
    "SP.POP.TOTL": ("population", "people"),
    "NY.GDP.MKTP.CD": ("gdp_current_usd", "current US dollars"),
    "SP.DYN.LE00.IN": ("life_expectancy", "years"),
    "IT.NET.USER.ZS": ("internet_users_pct", "% of population"),
}


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _get_json(url):
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        raw = response.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("Public data response exceeded the 3 MB limit.")
        return json.loads(raw.decode("utf-8"))


def _url(base, params):
    return base + "?" + urllib.parse.urlencode(params)


def _location(text, default="New Delhi"):
    patterns = (
        r"\b(?:weather|forecast|temperature|rain|wind|air quality|aqi|pollution)\s+(?:in|for|at)\s+([^?.,;]+)",
        r"\b(?:in|for|at)\s+([A-Z][A-Za-z .'-]{1,60})(?:\s+(?:today|tomorrow|now|currently|this week))?\s*[?.,;!]*$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            value = match.group(1).strip()
            value = re.sub(r"\s+(?:today|tomorrow|now|currently)$", "", value, flags=re.I)
            if value and len(value) <= 80:
                return value
    return default


def _country(text):
    match = re.search(r"\b(?:of|for|in)\s+([A-Z][A-Za-z .'-]{1,50})\s*[?.,;!]*$", text, re.I)
    value = match.group(1).strip() if match else "India"
    value = re.sub(r"\s+(?:population|gdp|economy|country profile)$", "", value, flags=re.I)
    return value or "India"


def _country_code(name):
    if re.fullmatch(r"[A-Za-z]{3}", name):
        return name.upper()
    url = "https://restcountries.com/v3.1/name/" + urllib.parse.quote(name, safe="") + "?" + urllib.parse.urlencode({"fullText": "true", "fields": "cca3,name"})
    payload = _get_json(url)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Could not resolve that country using REST Countries.")
    return str(payload[0].get("cca3") or "").upper()


def _weather(city, air=False):
    geo = _get_json(_url("https://geocoding-api.open-meteo.com/v1/search", {
        "name": city, "count": 1, "language": "en", "format": "json",
    }))
    places = geo.get("results") or []
    if not places:
        raise ValueError("Open-Meteo could not find the requested city.")
    place = places[0]
    if air:
        endpoint = "https://air-quality-api.open-meteo.com/v1/air-quality"
        fields = "pm10,pm2_5,european_aqi,us_aqi"
    else:
        endpoint = "https://api.open-meteo.com/v1/forecast"
        fields = "temperature_2m,precipitation,wind_speed_10m"
    payload = _get_json(_url(endpoint, {
        "latitude": place["latitude"], "longitude": place["longitude"],
        "hourly": fields, "forecast_days": 2, "timezone": "auto",
    }))
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    rows = []
    for index, stamp in enumerate(times[:24]):
        row = {"time_local": stamp}
        for field in fields.split(","):
            values = hourly.get(field) or []
            row[field] = values[index] if index < len(values) else None
        rows.append(row)
    is_air = bool(air)
    return {
        "dataset": "air_quality_forecast" if is_air else "weather_forecast",
        "source": "Open-Meteo Air Quality API" if is_air else "Open-Meteo Forecast API",
        "source_url": "https://open-meteo.com/en/docs/air-quality-api" if is_air else "https://open-meteo.com/en/docs",
        "api_url": endpoint,
        "retrieved_at_utc": _now(),
        "type": "model forecast, not certified station observations" if is_air else "weather model forecast, not a guarantee",
        "location": {"name": place.get("name"), "country": place.get("country"),
                     "latitude": place.get("latitude"), "longitude": place.get("longitude"),
                     "timezone": payload.get("timezone")},
        "units": payload.get("hourly_units", {}),
        "records": rows,
        "caveats": (["PM values are model estimates, not certified ground-station readings.",
                     "European AQI and US AQI are different scales; do not compare them directly.",
                     "Not an official public-health alert."] if is_air else
                    ["Forecast values change with model updates; they are not observed measurements."]),
    }


def _rates(text):
    codes = [CURRENCY_NAMES[word.casefold()] for word in re.findall(r"\b[A-Za-z]{3,8}\b", text)
             if word.casefold() in CURRENCY_NAMES]
    base = codes[0] if codes else "INR"
    url = _url("https://api.frankfurter.dev/v2/rates", {
        "base": base, "quotes": "USD,EUR,GBP,JPY,AUD,CAD,CHF,SGD,INR",
    })
    payload = _get_json(url)
    rows = [{"date": item.get("date"), "base": item.get("base"),
             "quote": item.get("quote"), "rate": item.get("rate"),
             "providers": item.get("providers")}
            for item in payload if isinstance(item, dict)]
    return {
        "dataset": "reference_exchange_rates", "source": "Frankfurter reference exchange-rate API",
        "source_url": "https://frankfurter.dev/", "api_url": url, "retrieved_at_utc": _now(),
        "type": "reference rates, not live trading quotes", "base_currency": base,
        "records": rows,
        "caveats": ["Rates may be delayed or revised and are not executable market prices.",
                    "Verify with a financial provider before financial, tax or accounting decisions."],
    }


def _earthquakes():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson"
    payload = _get_json(url)
    rows = []
    for feature in (payload.get("features") or [])[:40]:
        props = feature.get("properties") or {}
        coords = (feature.get("geometry") or {}).get("coordinates") or []
        rows.append({"event_id": feature.get("id"), "time_utc":
            datetime.fromtimestamp(props["time"] / 1000, timezone.utc).isoformat()
            if props.get("time") else None, "magnitude": props.get("mag"),
            "place": props.get("place"), "longitude": coords[0] if len(coords) > 0 else None,
            "latitude": coords[1] if len(coords) > 1 else None,
            "depth_km": coords[2] if len(coords) > 2 else None,
            "event_url": props.get("url"), "status": props.get("status")})
    return {
        "dataset": "earthquakes_m45_past_day", "source": "U.S. Geological Survey Earthquake Hazards Program",
        "source_url": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson",
        "api_url": url, "retrieved_at_utc": _now(), "type": "global earthquake event feed",
        "records": rows, "caveats": ["Event parameters can be revised as analysis continues.",
            "This magnitude 4.5+ feed is not a complete record of all earthquakes."],
    }


def _natural_events():
    url = _url("https://eonet.gsfc.nasa.gov/api/v3/events", {"status": "open", "limit": 50, "days": 30})
    payload = _get_json(url)
    rows = []
    for event in (payload.get("events") or [])[:30]:
        geometries = event.get("geometry") or []
        latest = geometries[-1] if geometries else {}
        rows.append({"event_id": event.get("id"), "title": event.get("title"),
            "description": event.get("description"), "date_utc": latest.get("date") if latest else None,
            "coordinates": latest.get("coordinates") if latest else None,
            "geometry_type": latest.get("type") if latest else None,
            "categories": [item.get("title") for item in event.get("categories", [])],
            "event_url": event.get("link"),
            "sources": [item.get("url") for item in event.get("sources", [])]})
    return {
        "dataset": "nasa_open_natural_events", "source": "NASA Earth Observatory Natural Event Tracker (EONET)",
        "source_url": "https://eonet.gsfc.nasa.gov/", "api_url": url, "retrieved_at_utc": _now(),
        "type": "curated open natural-event metadata", "records": rows,
        "caveats": ["EONET is not a complete authoritative hazard inventory.",
                    "Event geometry and status may be revised by curators."],
    }


def _country_profile(name):
    url = "https://restcountries.com/v3.1/name/" + urllib.parse.quote(name, safe="") + "?" + urllib.parse.urlencode({
        "fullText": "true", "fields": "name,cca3,capital,region,subregion,population,area,languages,currencies,timezones",
    })
    payload = _get_json(url)
    if not isinstance(payload, list) or not payload:
        raise ValueError("REST Countries returned no matching country profile.")
    item = payload[0]
    return {
        "dataset": "country_reference_profile", "source": "REST Countries",
        "source_url": "https://restcountries.com/", "api_url": url, "retrieved_at_utc": _now(),
        "type": "country reference metadata (not a live sensor feed)",
        "records": [{"country_code": item.get("cca3"),
            "name": (item.get("name") or {}).get("common"),
            "official_name": (item.get("name") or {}).get("official"),
            "capital": item.get("capital") or [], "region": item.get("region"),
            "subregion": item.get("subregion"), "population": item.get("population"),
            "area_km2": item.get("area"), "languages": item.get("languages") or {},
            "currencies": item.get("currencies") or {}, "timezones": item.get("timezones") or []}],
        "caveats": ["Population and area definitions can differ by source and update date."],
    }


def _world_bank(name):
    country = _country_code(name)
    start_year = max(1960, datetime.now(timezone.utc).year - 15)
    rows_by_year = {}
    indicator_info = []
    for code, (field, unit) in INDICATORS.items():
        url = _url("https://api.worldbank.org/v2/country/" + country + "/indicator/" + code, {
            "format": "json", "per_page": "100", "date": f"{start_year}:{datetime.now(timezone.utc).year}",
        })
        payload = _get_json(url)
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            raise ValueError("World Bank returned an unexpected response for " + code)
        for item in payload[1]:
            date = item.get("date")
            if date and str(date).isdigit():
                year = int(date)
                rows_by_year.setdefault(year, {"country_code": country, "year": year})[field] = item.get("value")
        indicator_info.append({"code": code, "field": field, "unit": unit})
    return {
        "dataset": "world_bank_economy", "source": "World Bank Open Data API",
        "source_url": "https://data.worldbank.org/", "api_url": "https://api.worldbank.org/v2/",
        "retrieved_at_utc": _now(), "type": "historical indicators; latest reported year may lag",
        "country_code": country, "indicators": indicator_info,
        "records": [rows_by_year[key] for key in sorted(rows_by_year)],
        "caveats": ["GDP is current USD, not inflation-adjusted.",
                    "Null values are missing observations, not zero.",
                    "Indicators are descriptive and do not establish causation."],
    }


def query_real_world_data(query: str) -> dict:
    """Retrieve one or more relevant live/public datasets for a natural-language request."""
    text = str(query or "").strip()
    lower = text.casefold()
    if not text or len(text) > 1000:
        raise ValueError("Enter a real-world data question of at most 1,000 characters.")
    jobs = []
    city = _location(text)
    country = _country(text)
    if any(term in lower for term in ("weather", "forecast", "temperature", "rain", "wind")):
        jobs.append(lambda: _weather(city, air=False))
    if any(term in lower for term in ("air quality", "aqi", "pm2.5", "pm2_5", "pollution")):
        jobs.append(lambda: _weather(city, air=True))
    if any(term in lower for term in ("exchange rate", "currency rate", "convert currency", "forex", "inr", "usd to", "eur to")):
        jobs.append(lambda: _rates(text))
    if any(term in lower for term in ("earthquake", "earthquakes", "seismic")):
        jobs.append(_earthquakes)
    if any(term in lower for term in ("natural event", "wildfire", "volcano", "storm event", "natural disaster", "eonet")):
        jobs.append(_natural_events)
    if any(term in lower for term in ("country profile", "capital of", "population of", "gdp of", "life expectancy", "internet users", "country data")):
        if any(term in lower for term in ("population", "gdp", "life expectancy", "internet users")):
            jobs.append(lambda: _world_bank(country))
        else:
            jobs.append(lambda: _country_profile(country))
    if not jobs:
        return {"matched": False, "message": "No specific public-data domain was identified.",
                "available_domains": ["weather", "air quality", "exchange rates", "earthquakes",
                                      "natural events", "country profiles", "World Bank indicators"]}
    datasets, failures = [], []
    for job in jobs[:4]:
        try:
            datasets.append(job())
        except Exception as exc:
            failures.append({"error": type(exc).__name__ + ": " + str(exc)})
    return {"matched": True, "query": text, "retrieved_at_utc": _now(),
            "datasets": datasets, "failures": failures,
            "status": "complete" if datasets and not failures else ("partial" if datasets else "failed")}
