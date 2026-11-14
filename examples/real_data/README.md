# Real-data examples for Orbit

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

Run the offline integrity check after generation:

    python examples/real_data/validate_data_lab.py

The validator checks artifact presence, row counts, source metadata and JSON/CSV consistency. It does not certify upstream scientific or financial accuracy.

Review `build_data_lab.py` and its generated manifest for source-specific terms and limitations before commercial reuse. In particular, Open-Meteo's free API terms may restrict commercial use; Frankfurter rates are reference rates, not trading quotes; USGS event parameters can be revised; NASA EONET is a curated feed, not a complete hazard inventory.
