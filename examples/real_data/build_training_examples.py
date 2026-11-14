#!/usr/bin/env python3
"""Convert a collected real-data lab snapshot into sourced FirstGPT examples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_INPUT = Path(__file__).resolve().parent / "generated" / "data_lab"
DEFAULT_TEXT = Path("data/raw/real_world_data_lab.txt")
DEFAULT_JSONL = Path("examples/real_data/generated/data_lab_qa_examples.jsonl")


def load_bundle(folder: Path, dataset_id: str):
    path = folder / (dataset_id + ".json")
    if not path.is_file():
        return None
    bundle = json.loads(path.read_text(encoding="utf-8"))
    return bundle if isinstance(bundle, dict) and isinstance(bundle.get("records"), list) else None


def make_example(question, answer, dataset_id, bundle, record, index):
    metadata = bundle.get("metadata") or {}
    return {
        "instruction": "Answer only from the supplied real-world record. Preserve the observation time, units, source and caveats. Do not invent missing values.",
        "input": question,
        "output": answer,
        "metadata": {
            "dataset_id": dataset_id,
            "record_index": index,
            "retrieved_at_utc": bundle.get("retrieved_at_utc"),
            "source": metadata.get("source"),
            "source_url": metadata.get("source_url"),
            "api_url": metadata.get("api_url"),
            "units": metadata.get("units"),
            "caveats": metadata.get("caveats", []),
            "record": record,
        },
    }


def build(folder: Path):
    examples = []
    caveats_by_kind = {
        "weather_forecast": "This is a model forecast, not an observed measurement.",
        "air_quality_forecast": "These are model estimates/forecasts, not certified ground-station measurements; AQI scales differ.",
        "earthquakes_m45_past_day": "This feed covers magnitude 4.5+ events from the past day; event parameters may be revised.",
        "inr_reference_exchange_rates": "These are reference rates, not live trading quotes.",
        "world_bank_economy": "These are published annual observations; null means missing, not zero. Current-USD GDP is not inflation-adjusted.",
        "country_reference_profile": "Country reference fields may be updated independently and may differ in definition from statistical series.",
        "nasa_open_natural_events": "NASA EONET is a curated event catalog, not a complete hazard inventory.",
    }

    bundle = load_bundle(folder, "world_bank_economy")
    if bundle:
        for index, row in enumerate(bundle["records"]):
            year = row.get("year")
            if year is None:
                continue
            for key, label, unit in (
                ("population", "population", "people"),
                ("gdp_current_usd", "GDP in current US dollars", "current US dollars"),
                ("life_expectancy", "life expectancy at birth", "years"),
                ("internet_users_pct", "internet use as a share of the population", "percent"),
            ):
                value = row.get(key)
                if value is None:
                    continue
                question = f"What was {label} for {row.get('country_code')} in {year} in the World Bank dataset?"
                answer = f"The World Bank dataset records {value} {unit} for {label} in {row.get('country_code')} in {year}. {caveats_by_kind['world_bank_economy']}"
                examples.append(make_example(question, answer, "world_bank_economy", bundle, row, index))

    bundle = load_bundle(folder, "weather_forecast")
    if bundle:
        for index, row in enumerate(bundle["records"][:24]):
            location = (bundle.get("metadata") or {}).get("location") or {}
            city = location.get("name") or "the requested location"
            question = f"What weather forecast values are recorded for {city} at {row.get('time_local')}?"
            answer = (
                f"For {city} at {row.get('time_local')}, the forecast records temperature "
                f"{row.get('temperature_2m_c')} °C, precipitation {row.get('precipitation_mm')} mm, "
                f"and wind speed {row.get('wind_speed_10m_kmh')} km/h. "
                + caveats_by_kind["weather_forecast"]
            )
            examples.append(make_example(question, answer, "weather_forecast", bundle, row, index))

    bundle = load_bundle(folder, "air_quality_forecast")
    if bundle:
        for index, row in enumerate(bundle["records"][:24]):
            location = (bundle.get("metadata") or {}).get("location") or {}
            city = location.get("name") or "the requested location"
            question = f"What air-quality model values are recorded for {city} at {row.get('time_local')}?"
            answer = (
                f"For {city} at {row.get('time_local')}, the model records PM2.5 {row.get('pm2_5')} µg/m³, "
                f"PM10 {row.get('pm10')} µg/m³, European AQI {row.get('european_aqi')}, "
                f"and US AQI {row.get('us_aqi')}. "
                + caveats_by_kind["air_quality_forecast"]
            )
            examples.append(make_example(question, answer, "air_quality_forecast", bundle, row, index))

    bundle = load_bundle(folder, "earthquakes_m45_past_day")
    if bundle:
        for index, row in enumerate(bundle["records"][:40]):
            question = f"What earthquake information is in USGS event {row.get('event_id')}?"
            answer = (
                f"USGS lists an event near {row.get('place')} at {row.get('time_utc')} UTC, "
                f"with magnitude {row.get('magnitude')}, depth {row.get('depth_km')} km, "
                f"and coordinates latitude {row.get('latitude')}, longitude {row.get('longitude')}. "
                + caveats_by_kind["earthquakes_m45_past_day"]
            )
            examples.append(make_example(question, answer, "earthquakes_m45_past_day", bundle, row, index))

    bundle = load_bundle(folder, "inr_reference_exchange_rates")
    if bundle:
        for index, row in enumerate(bundle["records"]):
            question = f"What INR reference exchange rate is recorded for {row.get('date')} against {row.get('quote')}?"
            answer = (
                f"Frankfurter records {row.get('rate')} {row.get('quote')} per 1 {row.get('base')} "
                f"on {row.get('date')}. " + caveats_by_kind["inr_reference_exchange_rates"]
            )
            examples.append(make_example(question, answer, "inr_reference_exchange_rates", bundle, row, index))

    bundle = load_bundle(folder, "country_reference_profile")
    if bundle and bundle["records"]:
        row = bundle["records"][0]
        country = row.get("name") or row.get("country_code") or "the selected country"
        question = f"What does the country reference profile report about {country}?"
        answer = (
            f"The profile identifies {country} (code {row.get('country_code')}), capital {row.get('capital')}, "
            f"region {row.get('region')}, population {row.get('population')}, area {row.get('area_km2')} km², "
            f"languages {json.dumps(row.get('languages'), ensure_ascii=False)}, "
            f"currencies {json.dumps(row.get('currencies'), ensure_ascii=False)}, "
            f"and time zones {json.dumps(row.get('timezones'), ensure_ascii=False)}. "
            + caveats_by_kind["country_reference_profile"]
        )
        examples.append(make_example(question, answer, "country_reference_profile", bundle, row, 0))

    bundle = load_bundle(folder, "nasa_open_natural_events")
    if bundle:
        for index, row in enumerate(bundle["records"][:40]):
            question = f"What does NASA EONET list for natural event {row.get('title')}?"
            answer = (
                f"NASA EONET lists {row.get('title')}, categories {json.dumps(row.get('categories'), ensure_ascii=False)}, "
                f"latest geometry date {row.get('date_utc')}, geometry type {row.get('geometry_type')}, "
                f"and coordinates {json.dumps(row.get('coordinates'), ensure_ascii=False)}. "
                + caveats_by_kind["nasa_open_natural_events"]
            )
            examples.append(make_example(question, answer, "nasa_open_natural_events", bundle, row, index))


    # Practical applied questions are generated from the exact retrieved records.
    # They are not fabricated measurements: each row retains source/time/units/caveats.
    bundle = load_bundle(folder, "inr_reference_exchange_rates")
    if bundle:
        for index, row in enumerate(bundle["records"]):
            rate = row.get("rate")
            if isinstance(rate, (int, float)) and row.get("base") == "INR" and row.get("quote"):
                amount = 5000
                converted = amount * rate
                examples.append(make_example(
                    f"A person has ₹{amount:,}. Using the supplied INR/{row.get('quote')} reference rate, approximately how much is that?",
                    f"The record gives 1 INR = {rate} {row.get('quote')}. So {amount:,} × {rate} = {converted:.2f} {row.get('quote')}. This is an indicative reference-rate calculation, not a bank/card quote; fees and spreads may apply.",
                    "inr_reference_exchange_rates", bundle, row, index))

    bundle = load_bundle(folder, "world_bank_economy")
    if bundle:
        rows = sorted(bundle["records"], key=lambda x: x.get("year") or 0)
        for field, label, unit in (("population", "population", "people"), ("internet_users_pct", "internet use", "percentage points")):
            valid = [r for r in rows if isinstance(r.get(field), (int, float))]
            if len(valid) >= 2:
                first, last = valid[0], valid[-1]
                delta = last[field] - first[field]
                delta_text = f"{delta:,.0f} people" if field == "population" else f"{delta:.2f} percentage points"
                examples.append(make_example(
                    f"Compare the earliest and latest available {label} observations for {first.get('country_code')} in this World Bank extract.",
                    f"The records show {first[field]} {unit} in {first.get('year')} and {last[field]} {unit} in {last.get('year')}. The arithmetic difference is {delta_text}. This describes the observations and does not establish why the value changed; missing years are not zero.",
                    "world_bank_economy", bundle, {"first": first, "last": last}, 0))

    bundle = load_bundle(folder, "country_reference_profile")
    if bundle and bundle["records"]:
        row = bundle["records"][0]
        population, area = row.get("population"), row.get("area_km2")
        if isinstance(population, (int, float)) and isinstance(area, (int, float)) and area > 0:
            examples.append(make_example(
                f"Calculate the approximate population density of {row.get('name') or 'this country'} from the supplied profile.",
                f"Population is {population:,} and area is {area:,.2f} km². Density = population ÷ area = {population / area:,.2f} people per km². This national average does not describe local distribution.",
                "country_reference_profile", bundle, row, 0))

    bundle = load_bundle(folder, "weather_forecast")
    if bundle:
        location = (bundle.get("metadata") or {}).get("location") or {}
        city = location.get("name") or "the requested location"
        row = next((r for r in bundle["records"] if "15:00" in str(r.get("time_local", ""))), None)
        if row:
            examples.append(make_example(
                f"I'm planning an outdoor activity around 3 pm in {city}. What does this forecast say?",
                f"For {city} at {row.get('time_local')}, the forecast gives {row.get('temperature_2m_c')} °C, precipitation {row.get('precipitation_mm')} mm and wind {row.get('wind_speed_10m_kmh')} km/h. Use these as planning inputs, recheck the latest forecast and local alerts, and remember forecasts are not guarantees.",
                "weather_forecast", bundle, row, bundle["records"].index(row)))

    bundle = load_bundle(folder, "air_quality_forecast")
    if bundle:
        location = (bundle.get("metadata") or {}).get("location") or {}
        city = location.get("name") or "the requested location"
        row = next((r for r in bundle["records"] if r.get("pm2_5") is not None), None)
        if row:
            examples.append(make_example(
                f"Help me interpret this air-quality forecast for {city}.",
                f"At {row.get('time_local')}, model output is PM2.5 {row.get('pm2_5')} µg/m³, PM10 {row.get('pm10')} µg/m³, European AQI {row.get('european_aqi')} and US AQI {row.get('us_aqi')}. PM values are particulate concentrations; the two AQI scales are not interchangeable. This is a model forecast, not a certified station reading or official health alert.",
                "air_quality_forecast", bundle, row, bundle["records"].index(row)))

    bundle = load_bundle(folder, "earthquakes_m45_past_day")
    if bundle and bundle["records"]:
        row = bundle["records"][0]
        examples.append(make_example(
            f"Explain the fields in this USGS earthquake record without overstating the risk.",
            f"USGS lists magnitude {row.get('magnitude')} near {row.get('place')}, depth {row.get('depth_km')} km, at {row.get('time_utc')} UTC. Coordinates are latitude {row.get('latitude')}, longitude {row.get('longitude')}. These fields describe the event but alone cannot determine damage or personal danger; follow official local emergency guidance.",
            "earthquakes_m45_past_day", bundle, row, 0))

    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--text-output", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--jsonl-output", type=Path, default=DEFAULT_JSONL)
    args = parser.parse_args()

    examples = build(args.input.expanduser().resolve())
    if not examples:
        raise SystemExit("No examples were generated. Build the real-data lab first and check its manifest.")

    text_lines = [
        "REAL-WORLD DATA TRAINING EXAMPLES",
        "Generated from a retrieved FirstAI multi-domain data-lab snapshot.",
        "Every answer is grounded in the record and carries source metadata in the JSONL companion.",
        "Forecasts are not observations; reference rates are not trading quotes; missing values are not zero.",
        "",
    ]
    for item in examples:
        text_lines.extend([
            "Question: " + item["input"],
            "Answer: " + item["output"],
            "Source: " + str(item["metadata"].get("source_url")),
            "Retrieved at (UTC): " + str(item["metadata"].get("retrieved_at_utc")),
            "",
        ])

    args.text_output.parent.mkdir(parents=True, exist_ok=True)
    args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
    args.text_output.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    args.jsonl_output.write_text(
        "".join(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n" for item in examples),
        encoding="utf-8",
    )
    print(json.dumps({
        "examples": len(examples),
        "text_training_file": str(args.text_output),
        "provenance_jsonl": str(args.jsonl_output),
        "included_domains": sorted({item["metadata"]["dataset_id"] for item in examples}),
    }, indent=2))


if __name__ == "__main__":
    main()
