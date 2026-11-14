#!/usr/bin/env python3
"""Fetch a small, reproducible multi-domain real-world data pack.

Uses only Python's standard library. Each source is independent: partial success
is preserved and recorded in manifest.json rather than hiding source failures.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API_TIMEOUT = 25
MAX_RESPONSE_BYTES = 20 * 1024 * 1024
INDICATORS = {
    "SP.POP.TOTL": ("population", "people"),
    "NY.GDP.MKTP.CD": ("gdp_current_usd", "current US dollars"),
    "SP.DYN.LE00.IN": ("life_expectancy", "years"),
    "IT.NET.USER.ZS": ("internet_users_pct", "% of population"),
}


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fetch_json(url):
    request = urllib.request.Request(url, headers={
        "User-Agent": "FirstAI-DataLab/1.0 (open-data educational project)",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(request, timeout=API_TIMEOUT) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("Source response exceeded the 20 MB safety limit.")
        return json.loads(raw.decode("utf-8"))


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def write_csv(path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["no_records"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in keys})


def world_bank(country, start_year, end_year):
    merged = {}
    indicators = []
    for code, (key, unit) in INDICATORS.items():
        query = urllib.parse.urlencode({"format": "json", "per_page": "100", "date": f"{start_year}:{end_year}"})
        url = f"https://api.worldbank.org/v2/country/{country}/indicator/{code}?{query}"
        payload = fetch_json(url)
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            raise ValueError(f"Unexpected World Bank response for {code}")
        for item in payload[1]:
            date = item.get("date")
            if not date or not str(date).isdigit():
                continue
            year = int(date)
            row = merged.setdefault(year, {"country_code": country, "year": year})
            row[key] = item.get("value")
        indicators.append({"code": code, "field": key, "unit": unit})
    return sorted(merged.values(), key=lambda row: row["year"]), {
        "source": "World Bank Open Data API",
        "source_url": "https://data.worldbank.org/",
        "api_url": "https://api.worldbank.org/v2/",
        "license_note": "World Bank indicators have dataset-specific terms; verify the relevant indicator metadata before commercial redistribution.",
        "caveats": ["GDP is current USD, not inflation-adjusted.", "Null values are missing observations, not zero.", "Indicators are descriptive and do not establish causation."],
        "indicators": indicators,
    }


def weather(city):
    geo_url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode({
        "name": city, "count": 1, "language": "en", "format": "json"
    })
    geo = fetch_json(geo_url)
    results = geo.get("results") or []
    if not results:
        raise ValueError("Open-Meteo could not find the requested city.")
    place = results[0]
    forecast_url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
        "latitude": place["latitude"], "longitude": place["longitude"],
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "forecast_days": 3, "timezone": "auto",
    })
    payload = fetch_json(forecast_url)
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    rows = []
    for index, stamp in enumerate(times):
        rows.append({
            "time_local": stamp,
            "temperature_2m_c": (hourly.get("temperature_2m") or [None] * len(times))[index],
            "precipitation_mm": (hourly.get("precipitation") or [None] * len(times))[index],
            "wind_speed_10m_kmh": (hourly.get("wind_speed_10m") or [None] * len(times))[index],
        })
    return rows, {
        "source": "Open-Meteo",
        "source_url": "https://open-meteo.com/",
        "api_url": "https://api.open-meteo.com/v1/forecast",
        "license_note": "Free Open-Meteo API use is subject to its applicable non-commercial terms; review current terms before commercial use.",
        "caveats": ["Forecast values are model output, not observations.", "Forecast changes as new model runs are published."],
        "location": {"name": place.get("name"), "country": place.get("country"),
                     "latitude": place.get("latitude"), "longitude": place.get("longitude"),
                     "timezone": payload.get("timezone")},
        "units": payload.get("hourly_units", {}),
    }


def earthquakes():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson"
    payload = fetch_json(url)
    rows = []
    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        coords = (feature.get("geometry") or {}).get("coordinates") or [None, None, None]
        rows.append({
            "event_id": feature.get("id"), "time_utc": datetime.fromtimestamp((props.get("time") or 0) / 1000, timezone.utc).isoformat() if props.get("time") else None,
            "magnitude": props.get("mag"), "place": props.get("place"), "longitude": coords[0],
            "latitude": coords[1], "depth_km": coords[2], "event_url": props.get("url"),
            "status": props.get("status"), "tsunami_flag": props.get("tsunami"),
        })
    return rows, {
        "source": "U.S. Geological Survey (USGS) Earthquake Hazards Program",
        "source_url": "https://earthquake.usgs.gov/earthquakes/feed/",
        "api_url": url,
        "license_note": "U.S. federal government data; credit USGS when using or redistributing.",
        "caveats": ["This feed contains magnitude 4.5+ events from the past day.", "Event parameters may be revised as the catalog is reviewed."],
        "feed_generated_at_utc": payload.get("metadata", {}).get("generated"),
    }


def exchange_rates():
    url = "https://api.frankfurter.dev/v2/rates?" + urllib.parse.urlencode({
        "base": "INR", "quotes": "USD,EUR,GBP,JPY,AUD,CAD,CHF,SGD"
    })
    payload = fetch_json(url)
    rows = [{
        "date": item.get("date"), "base": item.get("base"), "quote": item.get("quote"),
        "rate": item.get("rate"), "providers": item.get("providers"),
    } for item in payload if isinstance(item, dict)]
    return rows, {
        "source": "Frankfurter reference exchange-rate API",
        "source_url": "https://frankfurter.dev/",
        "api_url": "https://api.frankfurter.dev/v2/rates",
        "license_note": "Reference rates are sourced from central banks and official institutions; provider-specific terms may apply.",
        "caveats": ["These are reference rates, not live trading quotes.", "Rates may be delayed, missing or revised; verify with the provider for financial, tax or accounting decisions."],
        "base_currency": "INR",
    }


def country_profile(country):
    url = "https://restcountries.com/v3.1/alpha/" + urllib.parse.quote(country) + "?" + urllib.parse.urlencode({
        "fields": "name,cca3,capital,region,subregion,population,area,languages,currencies,timezones,flags"
    })
    payload = fetch_json(url)
    if not isinstance(payload, list) or not payload:
        raise ValueError("REST Countries returned no country profile.")
    item = payload[0]
    return [{
        "country_code": item.get("cca3"),
        "name": (item.get("name") or {}).get("common"),
        "official_name": (item.get("name") or {}).get("official"),
        "capital": ", ".join(item.get("capital") or []),
        "region": item.get("region"), "subregion": item.get("subregion"),
        "population": item.get("population"), "area_km2": item.get("area"),
        "languages": item.get("languages") or {},
        "currencies": item.get("currencies") or {},
        "timezones": item.get("timezones") or [],
        "flag_svg": (item.get("flags") or {}).get("svg"),
    }], {
        "source": "REST Countries",
        "source_url": "https://restcountries.com/",
        "api_url": url,
        "license_note": "Review REST Countries project terms and data-source attributions before commercial redistribution.",
        "caveats": ["Country profile values may be updated independently of the World Bank series.", "Population and area definitions can differ by source and update date."],
    }


def natural_events():
    url = "https://eonet.gsfc.nasa.gov/api/v3/events?" + urllib.parse.urlencode({
        "status": "open", "limit": 100, "days": 30
    })
    payload = fetch_json(url)
    rows = []
    for event in payload.get("events", []):
        geometry = event.get("geometry") or []
        latest = geometry[-1] if geometry else {}
        coordinates = (latest.get("coordinates") or []) if isinstance(latest, dict) else []
        rows.append({
            "event_id": event.get("id"), "title": event.get("title"),
            "description": event.get("description"), "date_utc": latest.get("date") if latest else None,
            "geometry_type": latest.get("type") if latest else None,
            "coordinates": coordinates, "categories": [item.get("title") for item in event.get("categories", [])],
            "closed": event.get("closed"), "event_url": event.get("link"),
            "sources": [item.get("url") for item in event.get("sources", [])],
        })
    return rows, {
        "source": "NASA Earth Observatory Natural Event Tracker (EONET)",
        "source_url": "https://eonet.gsfc.nasa.gov/",
        "api_url": url,
        "license_note": "EONET event metadata is subject to NASA EONET's disclaimer and source-specific terms.",
        "caveats": ["EONET is a curated reference of natural-event metadata, not a complete authoritative hazard inventory.", "Location geometry and event status may be revised by curators."],
    }


def render_dashboard(manifest, datasets, output):
    sections = []
    for dataset_id, entry in datasets.items():
        rows = entry.get("rows", [])
        fields = list(rows[0].keys()) if rows else []
        header = "".join(f"<th>{html.escape(str(field))}</th>" for field in fields[:8])
        body = []
        for row in rows[:12]:
            cells = "".join(f"<td>{html.escape(str(csv_value(row.get(field))))[:180]}</td>" for field in fields[:8])
            body.append("<tr>" + cells + "</tr>")
        source = entry.get("metadata", {}).get("source_url", "")
        sections.append(
            f"<section><h2>{html.escape(dataset_id.replace('_',' ').title())}</h2>"
            f"<p>{len(rows)} records · {html.escape(entry.get('metadata',{}).get('source',''))} · "
            f"<a href=\"{html.escape(source, quote=True)}\" rel=\"noopener\">Source</a></p>"
            + ("<div class=\"table-wrap\"><table><thead><tr>" + header + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>" if rows else "<p>No rows returned.</p>")
            + "</section>"
        )
    cards = "".join(
        f"<article><strong>{html.escape(key.replace('_',' ').title())}</strong><b>{entry.get('row_count',0)}</b><small>{html.escape(entry.get('status','unknown'))}</small></article>"
        for key, entry in manifest["datasets"].items()
    )
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FirstAI · Real-world Data Lab</title><style>
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif;background:#0c1118;color:#e8eef5}body{margin:0;padding:24px;max-width:1400px;margin:auto}header{padding:24px 0}h1{font-size:clamp(26px,5vw,42px);margin:0 0 8px}p,small{color:#9eafbf}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}.cards article,section{background:#141d27;border:1px solid #263646;border-radius:12px;padding:16px}.cards b{display:block;font-size:30px;margin:8px 0}section{margin-top:16px;overflow:hidden}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:9px;border-bottom:1px solid #2a3948;white-space:nowrap}th{color:#b9c9d8}a{color:#79c7ff}footer{padding:24px 0;color:#9eafbf;font-size:12px}
</style></head><body><header><h1>Real-world Data Lab</h1><p>Live source retrieval · per-source provenance · missing values preserved · generated """ + html.escape(manifest["retrieved_at_utc"]) + """</p></header><div class="cards">""" + cards + """</div>""" + "".join(sections) + """<footer>Data is provided by the listed source organizations. Check each source's terms, definitions, update cadence and limitations before relying on it.</footer></body></html>"""
    (output / "dashboard.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="IND", help="World Bank ISO-3 country code (default: IND)")
    parser.add_argument("--city", default="New Delhi", help="City for Open-Meteo forecast (default: New Delhi)")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=datetime.now(timezone.utc).year)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "generated" / "data_lab")
    args = parser.parse_args()
    country = args.country.upper()
    if not re.fullmatch(r"[A-Z]{3}", country):
        parser.error("--country must be a three-letter ISO-3 code such as IND or USA")
    if not args.city.strip() or len(args.city) > 100:
        parser.error("--city must contain 1 to 100 characters")
    if args.start_year < 1960 or args.end_year < args.start_year or args.end_year > datetime.now(timezone.utc).year + 1:
        parser.error("Choose a valid year range from 1960 through next year.")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    collectors = {
        "world_bank_economy": lambda: world_bank(country, args.start_year, args.end_year),
        "weather_forecast": lambda: weather(args.city.strip()),
        "earthquakes_m45_past_day": earthquakes,
        "inr_reference_exchange_rates": exchange_rates,
        "nasa_open_natural_events": natural_events,
        "country_reference_profile": lambda: country_profile(country),
    }
    manifest = {
        "title": "FirstAI multi-domain real-world data lab",
        "version": 1, "retrieved_at_utc": utc_now(),
        "parameters": {"country": country, "city": args.city.strip(),
                       "start_year": args.start_year, "end_year": args.end_year},
        "datasets": {},
        "missing_value_policy": "Null/blank means unavailable or unreported; it is never silently converted to zero.",
        "partial_success_allowed": True,
    }
    catalog = {}
    for dataset_id, collector in collectors.items():
        print(f"Fetching {dataset_id}…", flush=True)
        entry = {"status": "failed", "row_count": 0, "rows_file": dataset_id + ".json"}
        try:
            rows, metadata = collector()
            if not isinstance(rows, list):
                raise TypeError("Collector did not return a list of rows.")
            entry.update({"status": "success", "row_count": len(rows), "metadata": metadata,
                          "retrieved_at_utc": utc_now()})
            catalog[dataset_id] = {"metadata": metadata, "rows": rows}
            write_json(output / (dataset_id + ".json"), {"metadata": metadata, "retrieved_at_utc": entry["retrieved_at_utc"], "records": rows})
            write_csv(output / (dataset_id + ".csv"), rows)
            print(f"  {len(rows)} rows", flush=True)
        except Exception as exc:
            entry.update({"error": f"{type(exc).__name__}: {exc}", "metadata": {}})
            print(f"  unavailable: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        manifest["datasets"][dataset_id] = entry
    successes = sum(1 for entry in manifest["datasets"].values() if entry["status"] == "success")
    manifest["success_count"] = successes
    manifest["failed_count"] = len(collectors) - successes
    manifest["status"] = "complete" if successes == len(collectors) else ("partial" if successes else "failed")
    if catalog:
        render_dashboard(manifest, catalog, output)
    write_json(output / "manifest.json", manifest)
    write_json(output / "catalog.json", {
        "title": manifest["title"], "retrieved_at_utc": manifest["retrieved_at_utc"],
        "dataset_ids": list(catalog), "manifest_file": "manifest.json",
    })
    print(json.dumps({"status": manifest["status"], "success_count": successes,
                      "failed_count": manifest["failed_count"], "output": str(output)}, indent=2))
    if not successes:
        raise SystemExit("No sources succeeded. See manifest.json for per-source errors.")


if __name__ == "__main__":
    main()
