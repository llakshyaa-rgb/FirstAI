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
