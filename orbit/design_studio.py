"""Orbit's multi-format design and document export engine."""
from __future__ import annotations
import csv
import html
import json
import re
import uuid
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

OUTPUT_DIR = Path(__file__).resolve().parent / "generated"
SUPPORTED = {"pdf", "docx", "pptx", "xlsx", "csv", "json", "html", "md", "txt", "svg"}
THEMES = {
    "midnight": ("#101820", "#192633", "#edf3f8", "#a9bac8", "#d7b77a"),
    "paper": ("#f7f3eb", "#ffffff", "#242a30", "#65717b", "#9a663b"),
    "ocean": ("#edf4f7", "#ffffff", "#173044", "#526c7b", "#247c91"),
    "editorial": ("#f4f0e8", "#fffdf8", "#27231f", "#766d62", "#a33c32"),
}

def parse_markdown(source):
    blocks, paragraph, bullets, table, code = [], [], [], [], []
    in_code = False
    def flush_para():
        if paragraph:
            blocks.append(("paragraph", " ".join(x.strip() for x in paragraph)))
            paragraph.clear()
    def flush_bullets():
        if bullets:
            blocks.append(("bullets", list(bullets)))
            bullets.clear()
    def flush_table():
        if table:
            rows = []
            for line in table:
                cells = [x.strip() for x in line.strip().strip("|").split("|")]
                if cells and all(re.fullmatch(r":?-{3,}:?", x.replace(" ", "")) for x in cells):
                    continue
                rows.append(cells)
            if rows: blocks.append(("table", rows))
            table.clear()
    for line in str(source).replace("\r", "").split("\n"):
        s = line.strip()
        if s.startswith("~~~"):
            flush_para(); flush_bullets(); flush_table()
            if in_code:
                blocks.append(("code", "\n".join(code))); code.clear(); in_code = False
            else: in_code = True
            continue
        if in_code:
            code.append(line); continue
        if not s:
            flush_para(); flush_bullets(); flush_table(); continue
        if s.startswith("|") and "|" in s[1:]:
            flush_para(); flush_bullets(); table.append(s); continue
        flush_table()
        heading = re.match(r"^(#{1,6})\s+(.+)$", s)
        if heading:
            flush_para(); flush_bullets(); blocks.append(("heading", (len(heading.group(1)), heading.group(2))))
        elif re.match(r"^[-*+]\s+|^\d+[.)]\s+", s):
            flush_para(); bullets.append(re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", s))
        elif s.startswith("> "):
            flush_para(); flush_bullets(); blocks.append(("quote", s[2:]))
        elif s in {"---", "***", "___"}:
            flush_para(); flush_bullets(); blocks.append(("rule", ""))
        else:
            flush_bullets(); paragraph.append(s)
    flush_para(); flush_bullets(); flush_table()
    if in_code: blocks.append(("code", "\n".join(code)))
    return blocks

def _inline(text):
    value = html.escape(str(text))
    value = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"\*(.+?)\*", r"<em>\1</em>", value)
    tick = chr(96)
    value = re.sub(tick + r"(.+?)" + tick, r"<code>\1</code>", value)
    return value

def _html(title, subtitle, blocks, theme):
    bg, panel, ink, muted, accent = theme
    parts = []
    for kind, value in blocks:
        if kind == "heading":
            n = min(4, value[0] + 1); parts.append(f"<h{n}>{_inline(value[1])}</h{n}>")
        elif kind == "paragraph": parts.append("<p>" + _inline(value) + "</p>")
        elif kind == "bullets": parts.append("<ul>" + "".join("<li>" + _inline(x) + "</li>" for x in value) + "</ul>")
        elif kind == "table":
            rows = []
            for i, row in enumerate(value):
                tag = "th" if i == 0 else "td"
                rows.append("<tr>" + "".join(f"<{tag}>{_inline(x)}</{tag}>" for x in row) + "</tr>")
            parts.append("<div class='table-wrap'><table>" + "".join(rows) + "</table></div>")
        elif kind == "quote": parts.append("<blockquote>" + _inline(value) + "</blockquote>")
        elif kind == "code": parts.append("<pre><code>" + html.escape(value) + "</code></pre>")
        elif kind == "rule": parts.append("<hr>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>
*{{box-sizing:border-box}}body{{margin:0;padding:42px 18px;background:{bg};color:{ink};font:16px/1.72 Inter,system-ui,sans-serif}}main{{max-width:860px;margin:auto;background:{panel};padding:clamp(24px,6vw,64px);border-radius:18px;box-shadow:0 20px 70px #0002}}h1{{font-size:clamp(2rem,6vw,3.4rem);line-height:1.1;letter-spacing:-.04em;margin:0 0 12px}}.subtitle{{color:{muted};border-bottom:2px solid {accent};padding-bottom:24px}}h2,h3,h4{{line-height:1.25;margin-top:2em;color:{accent}}}li{{padding:.18em 0}}blockquote{{margin:24px 0;padding:14px 20px;border-left:4px solid {accent};background:{bg}}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px 12px;border:1px solid {muted};text-align:left}}th{{background:{bg};color:{accent}}}pre{{white-space:pre-wrap;background:{bg};padding:16px;border-radius:10px}}footer{{margin-top:36px;color:{muted};font-size:12px}}@media print{{body{{padding:0}}main{{box-shadow:none}}}}</style></head><body><main><h1>{html.escape(title)}</h1>{'<p class="subtitle">'+html.escape(subtitle)+'</p>' if subtitle else ''}{''.join(parts)}<footer>Designed with Orbit</footer></main></body></html>"""

def _pdf(path, title, subtitle, blocks, theme):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, HRFlowable
    except ImportError as exc:
        raise RuntimeError("PDF export requires ReportLab. Install requirements-design.txt.") from exc
    styles = getSampleStyleSheet()
    accent = colors.HexColor(theme[4])
    styles.add(ParagraphStyle(name="OrbitTitle", parent=styles["Title"], fontSize=27, leading=32, textColor=colors.HexColor(theme[2]), spaceAfter=10))
    styles.add(ParagraphStyle(name="OrbitSub", parent=styles["Normal"], fontSize=11, leading=16, textColor=colors.HexColor(theme[3]), spaceAfter=20))
    styles.add(ParagraphStyle(name="OrbitH", parent=styles["Heading2"], fontSize=16, leading=20, textColor=accent, spaceBefore=15, spaceAfter=7))
    styles.add(ParagraphStyle(name="OrbitBody", parent=styles["BodyText"], fontSize=10, leading=15, spaceAfter=8))
    styles.add(ParagraphStyle(name="OrbitCode", parent=styles["Code"], fontSize=8, leading=11))
    story = [Paragraph(html.escape(title), styles["OrbitTitle"])]
    if subtitle: story.append(Paragraph(html.escape(subtitle), styles["OrbitSub"]))
    for kind, value in blocks:
        if kind == "heading": story.append(Paragraph(html.escape(value[1]), styles["OrbitH"]))
        elif kind == "paragraph": story.append(Paragraph(_inline(value), styles["OrbitBody"]))
        elif kind == "bullets":
            story.extend(Paragraph("• " + _inline(x), styles["OrbitBody"]) for x in value)
        elif kind == "table":
            data = [[Paragraph(_inline(x), styles["OrbitBody"]) for x in row] for row in value]
            t = Table(data, repeatRows=1)
            t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(theme[0])),("TEXTCOLOR",(0,0),(-1,0),accent),("GRID",(0,0),(-1,-1),.4,colors.HexColor(theme[3])),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
            story.extend([t, Spacer(1, 8)])
        elif kind == "quote": story.append(Paragraph("“" + _inline(value) + "”", styles["Quote"]))
        elif kind == "code": story.append(Preformatted(value, styles["OrbitCode"]))
        elif kind == "rule": story.append(HRFlowable(width="100%", color=accent, thickness=.8, spaceBefore=8, spaceAfter=8))
    def footer(canvas, doc):
        canvas.saveState(); w, _ = A4
        canvas.setStrokeColor(accent); canvas.line(18*mm, 16*mm, w-18*mm, 16*mm)
        canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor(theme[3]))
        canvas.drawString(18*mm, 10*mm, "DESIGNED WITH ORBIT"); canvas.drawRightString(w-18*mm, 10*mm, str(doc.page)); canvas.restoreState()
    SimpleDocTemplate(str(path), pagesize=A4, rightMargin=19*mm, leftMargin=19*mm, topMargin=19*mm, bottomMargin=24*mm, title=title, author="Orbit").build(story, onFirstPage=footer, onLaterPages=footer)

def _docx(path, title, subtitle, blocks, theme):
    try:
        from docx import Document
        from docx.shared import Inches, Pt, RGBColor
    except ImportError as exc:
        raise RuntimeError("DOCX export requires python-docx. Install requirements-design.txt.") from exc
    doc = Document(); sec = doc.sections[0]
    sec.top_margin = Inches(.72); sec.bottom_margin = Inches(.72)
    normal = doc.styles["Normal"]; normal.font.name = "Aptos"; normal.font.size = Pt(10)
    normal.font.color.rgb = RGBColor.from_string(theme[2].lstrip("#"))
    doc.add_heading(title, 0)
    if subtitle: doc.add_paragraph(subtitle, style="Subtitle")
    for kind, value in blocks:
        if kind == "heading": doc.add_heading(value[1], min(3, max(1, value[0])))
        elif kind == "paragraph": doc.add_paragraph(value)
        elif kind == "bullets":
            for x in value: doc.add_paragraph(x, style="List Bullet")
        elif kind == "table":
            table = doc.add_table(rows=0, cols=max(map(len, value))); table.style = "Light Shading Accent 1"
            for row in value:
                cells = table.add_row().cells
                for i, cell in enumerate(cells): cell.text = row[i] if i < len(row) else ""
        elif kind == "quote": doc.add_paragraph(value, style="Quote")
        elif kind == "code":
            p = doc.add_paragraph(); run = p.add_run(value); run.font.name = "Consolas"; run.font.size = Pt(8)
        elif kind == "rule": doc.add_paragraph("─" * 42)
    doc.core_properties.title = title; doc.core_properties.author = "Orbit"; doc.save(path)

def _pptx(path, title, subtitle, blocks, theme):
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor
    except ImportError as exc:
        raise RuntimeError("PowerPoint export requires python-pptx. Install requirements-design.txt.") from exc
    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    bg, _, ink, _, accent = theme
    def paint(slide): slide.background.fill.solid(); slide.background.fill.fore_color.rgb = RGBColor.from_string(bg.lstrip("#"))
    slide = prs.slides.add_slide(prs.slide_layouts[6]); paint(slide)
    box = slide.shapes.add_textbox(Inches(.8), Inches(1.6), Inches(11.7), Inches(2.3))
    p = box.text_frame.paragraphs[0]; p.text = title; p.font.size = Pt(36); p.font.bold = True; p.font.color.rgb = RGBColor.from_string(ink.lstrip("#"))
    if subtitle:
        sub = slide.shapes.add_textbox(Inches(.85), Inches(4), Inches(11), Inches(1))
        p = sub.text_frame.paragraphs[0]; p.text = subtitle; p.font.size = Pt(18); p.font.color.rgb = RGBColor.from_string(accent.lstrip("#"))
    frame = None
    for kind, value in blocks:
        if kind == "heading":
            slide = prs.slides.add_slide(prs.slide_layouts[6]); paint(slide)
            frame = slide.shapes.add_textbox(Inches(.8), Inches(.6), Inches(11.8), Inches(6)).text_frame
            p = frame.paragraphs[0]; p.text = value[1]; p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = RGBColor.from_string(accent.lstrip("#"))
        else:
            if frame is None:
                slide = prs.slides.add_slide(prs.slide_layouts[6]); paint(slide)
                frame = slide.shapes.add_textbox(Inches(.85), Inches(.8), Inches(11.6), Inches(5.8)).text_frame
            values = value if kind == "bullets" else [value] if kind in {"paragraph","quote","code"} else [" | ".join(row) for row in value] if kind == "table" else []
            for item in values:
                p = frame.add_paragraph(); p.text = ("• " if kind == "bullets" else "") + str(item)[:1500]
                p.font.size = Pt(16); p.font.color.rgb = RGBColor.from_string(ink.lstrip("#"))
    prs.save(path)

def _xlsx(path, title, blocks):
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise RuntimeError("Excel export requires openpyxl. Install requirements-design.txt.") from exc
    wb = Workbook(); ws = wb.active
    ws.title = re.sub(r"[\[\]:*?/\\]", "", title)[:31] or "Orbit"
    ws.append([title]); ws["A1"].font = Font(size=18, bold=True, color="FFFFFF"); ws["A1"].fill = PatternFill("solid", fgColor="203040")
    for kind, value in blocks:
        if kind == "heading":
            ws.append([]); ws.append([value[1]])
            ws.cell(ws.max_row,1).font = Font(size=13,bold=True,color="FFFFFF")
            ws.cell(ws.max_row,1).fill = PatternFill("solid",fgColor="31566B")
        elif kind == "table":
            for row in value:
                ws.append(row)
                if ws.max_row <= 4:
                    for cell in ws[ws.max_row]:
                        cell.font = Font(bold=True,color="FFFFFF"); cell.fill = PatternFill("solid",fgColor="31566B")
        elif kind == "bullets":
            for x in value: ws.append([x])
        elif kind in {"paragraph","quote","code"}: ws.append([value])
    for col in ws.columns:
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(48,max(12,max(len(str(x.value or "")) for x in col)+2))
    ws.freeze_panes = "A2"; wb.save(path)

def _csv(path, blocks):
    tables = [v for k,v in blocks if k == "table"]
    rows = tables[0] if tables else [[k.upper(), v[1] if k == "heading" else v] for k,v in blocks if k in {"heading","paragraph","quote"}]
    with path.open("w",encoding="utf-8-sig",newline="") as f: csv.writer(f).writerows(rows)

def _svg(path, title, subtitle, blocks, theme):
    lines = []
    for kind, value in blocks:
        if kind == "heading": lines.append(value[1])
        elif kind == "paragraph": lines.extend(re.findall(r".{1,64}(?:\s|$)", value) or [value[:64]])
        elif kind == "bullets": lines.extend("• " + x for x in value)
        elif kind == "quote": lines.append("“" + value + "”")
        elif kind == "table": lines.extend(" · ".join(row) for row in value[:8])
    lines = [x.strip()[:90] for x in lines if x.strip()][:22]
    bg, _, ink, muted, accent = theme; height = max(700,300+len(lines)*48)
    items = [f'<text x="76" y="128" font-size="54" font-weight="700" fill="{ink}">{xml_escape(title[:55])}</text>']
    if subtitle: items.append(f'<text x="80" y="180" font-size="23" fill="{muted}">{xml_escape(subtitle[:100])}</text>')
    items.append(f'<rect x="80" y="220" width="180" height="6" rx="3" fill="{accent}"/>')
    for i,line in enumerate(lines): items.append(f'<text x="84" y="{300+i*48}" font-size="23" fill="{ink}">{xml_escape(line)}</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}"><rect width="1200" height="{height}" fill="{bg}"/>'+"".join(items)+"</svg>",encoding="utf-8")

def create_artifact(title, content, file_format, subtitle="", theme_name="midnight", filename=""):
    title = str(title or "Orbit design").strip()[:200]; subtitle = str(subtitle or "").strip()[:300]
    content = str(content or "").strip(); fmt = str(file_format or "pdf").lower().lstrip(".")
    if fmt not in SUPPORTED: raise ValueError("Supported formats: " + ", ".join(sorted(SUPPORTED)))
    if not content: raise ValueError("Add some content before exporting.")
    if len(content) > 100000: raise ValueError("Content is limited to 100,000 characters per artifact.")
    theme = THEMES.get(str(theme_name).lower(), THEMES["midnight"]); blocks = parse_markdown(content)
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_-]+","-",str(filename or title)).strip("-_").lower() or "orbit-design"
    name = stem[:64] + "-" + uuid.uuid4().hex[:8] + "." + fmt; path = OUTPUT_DIR / name
    if fmt == "pdf": _pdf(path,title,subtitle,blocks,theme)
    elif fmt == "docx": _docx(path,title,subtitle,blocks,theme)
    elif fmt == "pptx": _pptx(path,title,subtitle,blocks,theme)
    elif fmt == "xlsx": _xlsx(path,title,blocks)
    elif fmt == "csv": _csv(path,blocks)
    elif fmt == "json":
        payload = {"title": title, "subtitle": subtitle,
                   "blocks": [{"type": kind, "content": value} for kind, value in blocks]}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elif fmt == "html": path.write_text(_html(title,subtitle,blocks,theme),encoding="utf-8")
    elif fmt == "md": path.write_text("# "+title+("\n\n"+subtitle if subtitle else "")+"\n\n"+content+"\n",encoding="utf-8")
    elif fmt == "txt": path.write_text(title+("\n"+subtitle if subtitle else "")+"\n\n"+re.sub(r"[*#]","",content)+"\n",encoding="utf-8")
    elif fmt == "svg": _svg(path,title,subtitle,blocks,theme)
    return {"name":name,"url":"/generated/"+name,"format":fmt,"size":path.stat().st_size,"title":title}
