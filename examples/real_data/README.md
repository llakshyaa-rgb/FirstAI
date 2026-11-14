# Real-data examples for Orbit

## Included real-world training examples

FirstAI now includes a small, checked-in seed dataset:
- `examples/real_data/seed_examples.jsonl` — three instruction/input/output examples with source, indicator code, reference year, unit, value and retrieval date in metadata.
- `data/raw/real_world_examples.txt` — readable versions already included in the starter FirstGPT corpus. The existing `prepare_corpus.py` collects this file automatically.

The seed records cover India's 2025 population (1,463,865,525 people), 2025 GDP (about USD 3.956 trillion in current dollars), and 2024 life expectancy at birth (72.235 years). They are sourced annual observations/estimates, not live readings. The population and GDP values are displayed by World Bank DataBank/indicator pages; life expectancy is shown on the World Bank India country page. Re-fetch before using these figures as the latest values in future answers.

Sources:
- https://databank.worldbank.org/reports.aspx?country=IND&source=2
- https://data.worldbank.org/indicator/NY.GDP.MKTP.CD?locations=IN
- https://data.worldbank.org/country/in

The World Bank builder below can refresh the four-indicator dataset and generate additional grounded Q&A examples from the latest non-null values.

## Build a multi-domain training set from actual retrieved records

First collect the seven-domain dataset, then turn its real rows into question/answer examples:

```bash
python examples/real_data/build_data_lab.py
python examples/real_data/build_training_examples.py
python prepare_corpus.py
```

The converter creates:
- `data/raw/real_world_data_lab.txt` — automatically included by the normal FirstGPT corpus preparation.
- `examples/real_data/generated/data_lab_qa_examples.jsonl` — structured examples retaining the exact source, retrieval timestamp, units, caveats and source record.

It generates examples from whatever providers succeeded in that collection: historical economy indicators, weather forecasts, air-quality forecasts, earthquakes, reference exchange rates, country profiles and NASA natural events. It never fabricates a missing provider or missing value. The text file is a training-oriented rendering; JSONL is the auditable provenance companion.

After reviewing the corpus, train/resume FirstGPT using the normal training command and the corpus/checkpoint rules in the root README. Refreshing records does not update an existing checkpoint by itself.




These are generated from published World Bank observations, not placeholder numbers.

Run from the repository root:

    python examples/real_data/build_examples.py

Use another ISO-3 country code with:

    python examples/real_data/build_examples.py --country USA

The standard-library builder retrieves four indicators from the World Bank API: total population (SP.POP.TOTL), GDP in current US dollars (NY.GDP.MKTP.CD), life expectancy at birth (SP.DYN.LE00.IN), and internet use as a percentage of population (IT.NET.USER.ZS).

It creates:
- world_bank_data.csv — source observations for data analysis.
- world_bank_data.json — observations plus provenance.
- manifest.json — source, retrieval timestamp, indicator codes, units and missing-value policy.
- dashboard.html — responsive data website with latest-value cards and historical charts.
- research_report.md — source-linked report with interpretation cautions.
- world_bank_data.xlsx — optional workbook if openpyxl is installed.

Generated files are placed in examples/real_data/generated/. Missing values remain blank/null, never silently replaced with zero. GDP is current USD, not inflation-adjusted. The indicators are descriptive and do not prove causation.

Sources:
- https://data.worldbank.org/
- https://api.worldbank.org/v2/

## Orbit build exercise

After generating the files, ask Orbit:

"Build a usable data-explorer app from examples/real_data/generated. Include year and indicator filters, a data table, charts, units, source links and retrieval date. Preserve missing values as missing; do not fabricate data."

Then use Preview app to inspect the running app. This exercises data ingestion, frontend design, coding, research/reporting and spreadsheet output against one consistent real dataset.

## Multi-domain live data lab

A second builder collects seven datasets from six independent public-data providers:

    python examples/real_data/build_data_lab.py

Defaults: India (IND), New Delhi, and the period 2000 through the current year. Change the city, country or year range with `--city`, `--country`, `--start-year` and `--end-year`. No API keys or extra Python packages are required.

| Source | Real-world domain | Coverage |
|---|---|---|
| World Bank | Economy and development | Population, GDP, life expectancy, internet use |
| Open-Meteo | Weather | Three-day hourly forecast |
| Open-Meteo Air Quality | Environment | PM2.5, PM10, European AQI and US AQI forecasts |
| USGS | Earthquakes | Magnitude 4.5+ events from the past day |
| Frankfurter | Currency reference data | INR against eight major currencies |
| NASA EONET | Natural events | Open event metadata from the last 30 days |
| REST Countries | Country reference | Population, area, capital, languages, currencies and time zones |

Each successful source gets JSON and CSV exports with source attribution, retrieval time, units or metadata, and caveats. A failed provider is recorded separately; successful datasets are retained. The generated `dashboard.html` provides a browsable overview.

For repeatable refreshes, keep each run as a timestamped snapshot and compare it with the previous collection:

    python examples/real_data/refresh_data_lab.py

The refresh script saves each run under `examples/real_data/generated/data_lab_history/`, retains collector logs, validates the snapshot, and writes `refresh_report.md` plus `refresh_report.json`. Failed sources are marked unavailable rather than treated as unchanged or as zero. Previous snapshots are not overwritten.

Orbit can also retrieve supported live facts directly in ordinary chat through read-only public APIs. Examples:

- `weather in Patna`
- `air quality in Delhi`
- `India GDP`
- `Japan inflation`
- `USD to INR exchange rate`
- `latest magnitude 4.5 earthquakes`
- `country profile for India`
- `NASA natural events`

The chat lookup returns source URLs, retrieval time, units where available and caveats. This is separate from training FirstGPT: live records are fetched at request time and are not automatically added to model weights or training data. It works without a Brave API key for the listed structured providers; general web research still needs its configured search provider.

Run the offline integrity check after generation:

    python examples/real_data/validate_data_lab.py

The validator checks artifact presence, row counts, source metadata and JSON/CSV consistency. It does not certify upstream scientific or financial accuracy.

Review `build_data_lab.py` and its generated manifest for source-specific terms and limitations before commercial reuse. In particular, Open-Meteo's free API terms may restrict commercial use; Frankfurter rates are reference rates, not trading quotes; USGS event parameters can be revised; NASA EONET is a curated feed, not a complete hazard inventory.
