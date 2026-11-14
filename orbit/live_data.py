"""Keyless live-data retrieval for Orbit using public, documented APIs.

These are bounded read-only lookups. Every result carries source URLs and a
retrieval timestamp; forecasts, reference rates and static metadata are labeled.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone


USER_AGENT = "FirstAI-Orbit/0.1 (read-only public data)"
TIMEOUT = 18


def _get_json(url: str):
    request = urllib.request.Request(url, headers={
        "Accept": "application/json", "User-Agent": USER_AGENT,
    })
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        raw = response.read(3_000_001)
    if len(raw) > 3_000_000:
        raise ValueError("The public data response exceeded the 3 MB limit.")
    return json.loads(raw.decode("utf-8"))


def _result(title, source, source_url, api_url, records, caveats=(), units=None, kind="current data"):
    return {
        "title": title, "data_type": kind,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source, "source_url": source_url, "api_url": api_url,
        "units": units or {}, "caveats": list(caveats), "records": records,
    }


def _city_from_query(query):
    match = re.search(r"\b(?:weather|temperature|forecast|air quality|aqi|pollution)\s+(?:in|for|at|of)\s+([A-Za-z][A-Za-z .'-]{1,70})", query, re.I)
    if match:
        return match.group(1).strip(" .?!")
    match = re.search(r"\b(?:in|for)\s+([A-Za-z][A-Za-z .'-]{1,50})\s+(?:weather|air quality|aqi)", query, re.I)
    return match.group(1).strip() if match else None


def _geocode(city):
    url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode({
        "name": city, "count": 1, "language": "en", "format": "json",
    })
    payload = _get_json(url)
    places = payload.get("results") or []
    if not places:
        raise ValueError("I couldn't resolve that place. Try a city name with its country.")
    return places[0], url


def _weather(query, air=False):
    city = _city_from_query(query)
    if not city:
        raise ValueError("Tell me the city, for example: 'weather in Patna' or 'air quality in Delhi'.")
    place, geo_url = _geocode(city)
    if air:
        endpoint = "https://air-quality-api.open-meteo.com/v1/air-quality"
        fields = "pm2_5,pm10,european_aqi,us_aqi"
        params = {"latitude": place["latitude"], "longitude": place["longitude"],
                  "hourly": fields, "forecast_days": 2, "timezone": "auto"}
        label, kind = "Air-quality forecast for " + place.get("name", city), "modeled forecast"
        caveats = ["Forecast/model estimates, not certified ground-station measurements.",
                   "European AQI and US AQI use different scales and are not interchangeable.",
                   "Not an official public-health alert."]
    else:
        endpoint = "https://api.open-meteo.com/v1/forecast"
        fields = "temperature_2m,precipitation,wind_speed_10m"
        params = {"latitude": place["latitude"], "longitude": place["longitude"],
                  "hourly": fields, "forecast_days": 2, "timezone": "auto"}
        label, kind = "Weather forecast for " + place.get("name", city), "modeled forecast"
        caveats = ["Forecast values are model output, not guaranteed observations."]
    api_url = endpoint + "?" + urllib.parse.urlencode(params)
    payload = _get_json(api_url)
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    records = []
    for index, stamp in enumerate(times[:36]):
        row = {"time_local": stamp}
        for field in fields.split(","):
            values = hourly.get(field) or []
            row[field] = values[index] if index < len(values) else None
        records.append(row)
    return _result(label, "Open-Meteo", "https://open-meteo.com/", api_url, records,
        caveats + ["Place resolved through " + geo_url],
        units=payload.get("hourly_units", {}), kind=kind)


def _earthquakes():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson"
    payload = _get_json(url)
    rows = []
    for feature in (payload.get("features") or [])[:50]:
        props = feature.get("properties") or {}
        coords = (feature.get("geometry") or {}).get("coordinates") or []
        rows.append({"id": feature.get("id"), "magnitude": props.get("mag"),
            "place": props.get("place"), "time_utc": datetime.fromtimestamp(
                props["time"] / 1000, timezone.utc).isoformat() if props.get("time") else None,
            "url": props.get("url"), "longitude": coords[0] if len(coords)>0 else None,
            "latitude": coords[1] if len(coords)>1 else None,
            "depth_km": coords[2] if len(coords)>2 else None})
    return _result("Magnitude 4.5+ earthquakes in the past day", "USGS",
        "https://earthquake.usgs.gov/earthquakes/feed/", url, rows,
        ["This feed covers magnitude 4.5+ events in the past day; event parameters may be revised."])


def _rates(query):
    pairs = {
        "usd": "USD", "dollar": "USD", "us dollar": "USD", "eur": "EUR",
        "euro": "EUR", "gbp": "GBP", "pound": "GBP", "jpy": "JPY",
        "yen": "JPY", "aud": "AUD", "cad": "CAD", "sgd": "SGD",
        "inr": "INR", "rupee": "INR",
    }
    codes = []
    lower = query.casefold()
    for key, code in pairs.items():
        if re.search(r"\b" + re.escape(key) + r"\b", lower) and code not in codes:
            codes.append(code)
    if len(codes) < 2:
        raise ValueError("Specify two currencies, e.g. 'USD to INR exchange rate'.")
    base, quote = codes[0], codes[1]
    url = "https://api.frankfurter.dev/v2/rates?" + urllib.parse.urlencode({
        "base": base, "quotes": quote,
    })
    rows = _get_json(url)
    return _result(base + " to " + quote + " reference exchange rate", "Frankfurter",
        "https://frankfurter.dev/", url, rows,
        ["Reference rate, not a live trading quote; the latest available business-day rate may be delayed."])


def _country(query):
    known = {
        "india": "ind", "united states": "usa", "usa": "usa", "us": "usa",
        "united kingdom": "gbr", "uk": "gbr", "japan": "jpn", "china": "chn",
        "germany": "deu", "france": "fra", "brazil": "bra", "canada": "can",
        "australia": "aus", "russia": "rus", "south korea": "kor",
        "south africa": "zaf", "nepal": "npl", "pakistan": "pak",
        "bangladesh": "bgd", "sri lanka": "lka", "indonesia": "idn",
        "mexico": "mex", "italy": "ita", "spain": "esp",
    }
    lower = query.casefold()
    match = next(((name, code) for name, code in sorted(known.items(), key=lambda x: -len(x[0])) if re.search(r"\b" + re.escape(name) + r"\b", lower)), None)
    if not match:
        raise ValueError("Name a country, for example: 'country profile for India'.")
    name, code = match
    url = "https://restcountries.com/v3.1/alpha/" + code + "?fields=name,cca3,capital,region,subregion,population,area,languages,currencies,timezones"
    payload = _get_json(url)
    item = payload[0] if isinstance(payload, list) and payload else payload
    row = {"name": (item.get("name") or {}).get("common"),
        "official_name": (item.get("name") or {}).get("official"), "code": item.get("cca3"),
        "capital": item.get("capital"), "region": item.get("region"),
        "subregion": item.get("subregion"), "population": item.get("population"),
        "area_km2": item.get("area"), "languages": item.get("languages"),
        "currencies": item.get("currencies"), "timezones": item.get("timezones")}
    return _result("Country profile: " + name.title(), "REST Countries",
        "https://restcountries.com/", url, [row],
        ["Population and area values depend on source definitions and update schedules."],
        kind="reference metadata")


def _nasa_events():
    url = "https://eonet.gsfc.nasa.gov/api/v3/events?" + urllib.parse.urlencode({
        "status": "open", "limit": 50, "days": 30,
    })
    payload = _get_json(url)
    rows = []
    for event in (payload.get("events") or [])[:50]:
        geometry = event.get("geometry") or []
        latest = geometry[-1] if geometry else {}
        rows.append({"id": event.get("id"), "title": event.get("title"),
            "description": event.get("description"),
            "latest_geometry_date": latest.get("date") if isinstance(latest, dict) else None,
            "categories": [x.get("title") for x in event.get("categories", [])],
            "source_url": event.get("sources", [{}])[0].get("url") if event.get("sources") else None})
    return _result("Open natural events in the last 30 days", "NASA EONET",
        "https://eonet.gsfc.nasa.gov/", url, rows,
        ["EONET catalogs natural events; it is not a complete emergency alert service."])


def _world_bank(query):
    lower = query.casefold()
    indicators = {
        "gdp per capita": "NY.GDP.PCAP.CD", "gdp": "NY.GDP.MKTP.CD",
        "inflation": "FP.CPI.TOTL.ZG", "population": "SP.POP.TOTL",
        "unemployment": "SL.UEM.TOTL.ZS", "life expectancy": "SP.DYN.LE00.IN",
    }
    indicator = next(((label, code) for label, code in sorted(indicators.items(), key=lambda x: -len(x[0])) if label in lower), None)
    countries = {"india":"IND","usa":"USA","united states":"USA","china":"CHN","japan":"JPN",
        "germany":"DEU","united kingdom":"GBR","uk":"GBR","france":"FRA","brazil":"BRA",
        "canada":"CAN","australia":"AUS","pakistan":"PAK","bangladesh":"BGD","nepal":"NPL"}
    country = next(((name, code) for name, code in sorted(countries.items(), key=lambda x:-len(x[0])) if re.search(r"\b"+re.escape(name)+r"\b",lower)),None)
    if not indicator or not country:
        raise ValueError("Specify an indicator and country, e.g. 'India GDP' or 'Japan inflation'. Supported indicators: GDP, GDP per capita, inflation, population, unemployment and life expectancy.")
    label, ind_code = indicator
    name, country_code = country
    url = "https://api.worldbank.org/v2/country/" + country_code + "/indicator/" + ind_code + "?" + urllib.parse.urlencode({"format":"json","per_page":8})
    payload = _get_json(url)
    rows = []
    if isinstance(payload, list) and len(payload) > 1:
        for item in payload[1] or []:
            if item and item.get("value") is not None:
                rows.append({"country":name.title(),"indicator":label,"year":item.get("date"),
                    "value":item.get("value"),"unit":(item.get("unit") or item.get("indicator",{}).get("id"))})
    return _result(name.title()+" "+label+" (World Bank)", "World Bank",
        "https://data.worldbank.org/", url, rows,
        ["Annual indicator data; the newest published year may lag the current date."])


def fetch_live_data(query: str) -> dict:
    """Resolve a supported live-data question to one authoritative public API."""
    text = str(query or "").strip()
    if not text or len(text) > 500:
        raise ValueError("Enter a live-data question up to 500 characters.")
    lower = text.casefold()
    if any(term in lower for term in ("air quality", "aqi", "pm2.5", "pm10", "pollution")):
        return _weather(text, air=True)
    if any(term in lower for term in ("weather", "temperature", "forecast", "rain", "wind")):
        return _weather(text)
    if any(term in lower for term in ("earthquake", "seismic", "quake")):
        return _earthquakes()
    if any(term in lower for term in ("exchange rate", "currency", " to inr", " usd", " eur", " gbp")):
        return _rates(text)
    if any(term in lower for term in ("country profile", "capital of", "population of", "country data")):
        return _country(text)
    if any(term in lower for term in ("natural event", "wildfire", "volcano", "flood event", "hurricane", "cyclone", "eonet")):
        return _nasa_events()
    if any(term in lower for term in ("gdp", "inflation", "unemployment", "life expectancy", "world bank")):
        return _world_bank(text)
    raise ValueError("I can fetch weather, air quality, earthquakes, currency reference rates, country profiles, NASA natural-event catalogs, and World Bank indicators. Specify the place, country or currencies.")
