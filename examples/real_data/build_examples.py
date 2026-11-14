#!/usr/bin/env python3
"""Generate real-data demo artifacts from World Bank Open Data (stdlib only)."""
from __future__ import annotations
import argparse, csv, html, json, re, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

INDICATORS = {
    "population": ("SP.POP.TOTL", "Population, total", "people"),
    "gdp_current_usd": ("NY.GDP.MKTP.CD", "GDP (current US$)", "current US dollars"),
    "life_expectancy": ("SP.DYN.LE00.IN", "Life expectancy at birth, total (years)", "years"),
    "internet_users_pct": ("IT.NET.USER.ZS", "Individuals using the Internet (% of population)", "percent"),
}
def fetch(country, code):
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{code}?format=json&per_page=100"
    request = Request(url, headers={"User-Agent": "Orbit-Real-Data-Examples/1.0"})
    error = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
                raise RuntimeError("Unexpected API response for " + code)
            return [r for r in payload[1] if isinstance(r, dict) and r.get("value") is not None]
        except (URLError, TimeoutError, ValueError, RuntimeError) as exc:
            error = exc
            if attempt < 2: time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not retrieve {code}: {error}") from error

def build(country, output):
    output.mkdir(parents=True, exist_ok=True)
    retrieved = datetime.now(timezone.utc).isoformat(timespec="seconds")
    series = {}
    for key, (code, label, unit) in INDICATORS.items():
        series[key] = {str(r["date"]): r["value"] for r in fetch(country, code) if str(r.get("date", "")).isdigit()}
    years = sorted({year for values in series.values() for year in values}, key=int)
    records = [{"country_code": country, "year": int(year), **{key: series[key].get(year) for key in INDICATORS}} for year in years]
    manifest = {
        "country_code": country, "retrieved_at_utc": retrieved,
        "source": "World Bank Open Data API", "source_url": "https://data.worldbank.org/",
        "api_base": "https://api.worldbank.org/v2/",
        "indicators": [{"key": k, "code": c, "label": n, "unit": u} for k, (c,n,u) in INDICATORS.items()],
        "record_count": len(records),
        "missing_value_policy": "Null means no published observation; it is not zero."
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "world_bank_data.json").write_text(json.dumps({"manifest": manifest, "records": records}, indent=2), encoding="utf-8")
    fields = ["country_code", "year", *INDICATORS]
    with (output / "world_bank_data.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(records)

    def display(key, value):
        if value is None: return "No published observation"
        if key == "population": return f"{int(value):,}"
        if key == "gdp_current_usd": return f"USD {value / 1e9:,.2f} billion"
        if key == "internet_users_pct": return f"{value:.1f}%"
        return f"{value:.2f} years"
    latest = {}
    for key in INDICATORS:
        rows = [r for r in records if r[key] is not None]
        latest[key] = max(rows, key=lambda r:r["year"]) if rows else None
    cards = []
    for key, (_, label, _) in INDICATORS.items():
        row = latest[key]
        value = display(key, row[key]) if row else "No published observation"
        year = str(row["year"]) if row else ""
        cards.append("<article><span>" + html.escape(label) + "</span><strong>" + html.escape(value) + "</strong><small>" + year + "</small></article>")
    chart = {k:[{"year":r["year"],"value":r[k]} for r in records if r[k] is not None] for k in ("population","life_expectancy","internet_users_pct")}
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>World Bank explorer</title><style>
:root{color-scheme:dark;font:16px system-ui;background:#101820;color:#edf3f7}body{margin:auto;padding:clamp(18px,4vw,48px);max-width:1100px}p,small,span{color:#aab9c5}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}article,section{background:#1b2934;border:1px solid #344653;border-radius:14px;padding:18px}article span,article strong,article small{display:block}article strong{font-size:1.5rem;margin:12px 0;overflow-wrap:anywhere}section{margin-top:18px}svg{width:100%;height:220px}a{color:#8bd5ca}</style></head><body>
<h1>World Bank data explorer · """ + html.escape(country) + """</h1><p>Published observations retrieved from World Bank Open Data. Latest published values are shown with their year.</p><div class="cards">""" + "".join(cards) + """</div><section><h2>Historical indicators</h2><p>Each series is scaled independently. Missing observations are omitted, not converted to zero.</p><div id="charts"></div></section>
<p>Retrieved """ + html.escape(retrieved) + """ · Source: <a href="https://data.worldbank.org/">World Bank Open Data</a> · See manifest.json for codes and units.</p>
<script>const data=""" + json.dumps(chart) + """;
const names={population:"Population (people)",life_expectancy:"Life expectancy (years)",internet_users_pct:"Internet users (% of population)"};
for(const [key,rows] of Object.entries(data)){if(!rows.length)continue;const box=document.createElement("section"),h=document.createElement("h3");h.textContent=names[key];box.append(h);const w=900,H=190,p=24,v=rows.map(x=>Number(x.value)),lo=Math.min(...v),hi=Math.max(...v),span=hi-lo||1;const pts=rows.map((x,i)=>[p+i*(w-2*p)/Math.max(1,rows.length-1),H-p-(Number(x.value)-lo)*(H-2*p)/span]);const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");svg.setAttribute("viewBox","0 0 "+w+" "+H);const line=document.createElementNS(svg.namespaceURI,"polyline");line.setAttribute("points",pts.map(x=>x.join(",")).join(" "));line.setAttribute("fill","none");line.setAttribute("stroke","#70c7bd");line.setAttribute("stroke-width","3");svg.append(line);box.append(svg);document.querySelector("#charts").append(box);}
</script></body></html>"""
    (output / "dashboard.html").write_text(page, encoding="utf-8")
    report = ["# World Bank indicator report: " + country, "", "- Retrieved (UTC): " + retrieved,
              "- Source: https://data.worldbank.org/", "- API: https://api.worldbank.org/v2/", "",
              "| Indicator | Latest year | Value | Unit | Code |", "|---|---:|---:|---|---|"]
    for key,(code,label,unit) in INDICATORS.items():
        row=latest[key]; report.append(f"| {label} | {row['year'] if row else '—'} | {display(key,row[key]) if row else '—'} | {unit} | {code} |")
    report += ["", "## Interpretation and limitations", "", "Null values mean no published observation, not zero. GDP is current US dollars and is not adjusted for inflation. These descriptive indicators do not establish causation. Check source metadata before consequential decisions."]
    (output / "research_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    try:
        from openpyxl import Workbook
        book=Workbook(); sheet=book.active; sheet.title="World Bank data"; sheet.append(fields)
        for row in records: sheet.append([row.get(k) for k in fields])
        sheet.freeze_panes="A2"; sheet.auto_filter.ref=sheet.dimensions
        book.save(output / "world_bank_data.xlsx")
    except ImportError: pass
    # Turn retrieved observations into grounded instruction examples for FirstGPT.
    # Keep them separate from raw data so they can be reviewed before training.
    qa_examples = []
    training_lines = [
        "REAL-WORLD DATA EXAMPLES — World Bank Open Data",
        "Source: https://data.worldbank.org/ | API: https://api.worldbank.org/v2/",
        "Retrieved at (UTC): " + retrieved,
        "Country code: " + country,
        "All values below are published observations, not forecasts.",
        "Missing observations are unknown, never zero.",
        "",
    ]
    for key, (code, label, unit) in INDICATORS.items():
        available = [row for row in records if row.get(key) is not None]
        if not available:
            continue
        row = max(available, key=lambda item: item["year"])
        value = row[key]
        if key == "population":
            shown = f"{int(value):,} people"
        elif key == "gdp_current_usd":
            shown = f"USD {value:,.2f} (current dollars)"
        elif key == "internet_users_pct":
            shown = f"{value:.2f}% of the population"
        else:
            shown = f"{value:.2f} years"
        question = f"What was {label.lower()} for country {country} in {row['year']} according to the World Bank?"
        answer = f"The World Bank observation for {country} in {row['year']} was {shown}. This is the latest non-null year in this retrieved dataset for this indicator. Source: https://api.worldbank.org/v2/country/{country}/indicator/{code}?format=json."
        qa_examples.append({
            "instruction": "Answer using only the sourced observation. Preserve the year and unit, and do not infer causation.",
            "input": question,
            "output": answer,
            "metadata": {
                "country_code": country, "year": row["year"], "indicator": key,
                "indicator_code": code, "value": value, "unit": unit,
                "source": "World Bank Open Data API",
                "source_url": f"https://api.worldbank.org/v2/country/{country}/indicator/{code}?format=json",
                "retrieved_at_utc": retrieved,
                "data_kind": "historical_observation",
            },
        })
        training_lines.extend(["Question: " + question, "Answer: " + answer, ""])
    qa_path = output / "world_bank_qa_examples.jsonl"
    qa_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in qa_examples), encoding="utf-8")
    training_path = output / "world_bank_training_examples.txt"
    training_path.write_text("\n".join(training_lines) + "\n", encoding="utf-8")
    return {"output":str(output.resolve()),"records":len(records),"retrieved_at_utc":retrieved,
            "qa_examples":len(qa_examples),"files":sorted(p.name for p in output.iterdir())}

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country",default="IND",help="ISO-3 country code; default IND")
    parser.add_argument("--output",type=Path,default=Path(__file__).with_name("generated"))
    args=parser.parse_args()
    country = args.country.upper()
    if not re.fullmatch(r"[A-Z]{3}", country):
        parser.error("--country must be a three-letter ISO-3 code, for example IND or USA")
    print(json.dumps(build(country,args.output),indent=2))
