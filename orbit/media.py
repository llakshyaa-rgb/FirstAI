"""Orbit media intake, safe local storage, text extraction and web image search.

No uploaded file is executed. Image search uses Brave Search when a key is configured.
"""
from __future__ import annotations
import base64
import hashlib
from html.parser import HTMLParser
import json
import mimetypes
import os
import re
import secrets
import urllib.parse
import urllib.request
import zipfile
import tarfile
from email import policy
from email.parser import BytesParser
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".avif", ".heic", ".heif"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".jsonl", ".xml", ".html", ".htm", ".rtf", ".py", ".js", ".ts", ".tsx", ".jsx", ".css", ".sql", ".yaml", ".yml", ".toml", ".ini", ".log"}
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp", ".epub", ".rtf"}
MEDIA_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".aiff", ".aif", ".opus", ".amr", ".wma", ".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".3gp", ".mpeg", ".mpg", ".wmv", ".flv", ".ts"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | TEXT_EXTENSIONS | DOCUMENT_EXTENSIONS | MEDIA_EXTENSIONS | {".zip", ".tar", ".tgz", ".tar.gz"}

def _safe_name(name: str) -> str:
    name = Path(str(name or "upload")).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return (name or "upload")[:120]

def _image_signature(data: bytes, ext: str) -> bool:
    if ext in {".jpg", ".jpeg"}: return data.startswith(b"\xff\xd8\xff")
    if ext == ".png": return data.startswith(b"\x89PNG\r\n\x1a\n")
    if ext == ".gif": return data.startswith((b"GIF87a", b"GIF89a"))
    if ext == ".webp": return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    if ext == ".bmp": return data.startswith(b"BM")
    if ext in {".tif", ".tiff"}: return data.startswith((b"II*\x00", b"MM\x00*"))
    if ext == ".avif": return len(data) >= 12 and data[4:8] == b"ftyp" and b"avif" in data[8:16]
    if ext in {".heic", ".heif"}: return len(data) >= 12 and data[4:8] == b"ftyp" and any(tag in data[8:32] for tag in (b"heic", b"heix", b"mif1", b"msf1"))
    return True

def parse_multipart(content_type: str, body: bytes):
    if len(body) > MAX_UPLOAD_BYTES + 1024 * 1024:
        raise ValueError("Upload is larger than the 20 MB limit.")
    envelope = (f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n").encode("ascii") + body
    message = BytesParser(policy=policy.default).parsebytes(envelope)
    if not message.is_multipart():
        raise ValueError("Expected a multipart file upload.")
    found = []
    total = 0
    for part in message.iter_parts():
        filename = part.get_filename()
        if not filename:
            continue
        data = part.get_payload(decode=True) or b""
        total += len(data)
        if total > MAX_UPLOAD_BYTES:
            raise ValueError("The combined upload batch must not exceed 20 MB.")
        found.append((_safe_name(filename), data, part.get_content_type()))
    if not found:
        raise ValueError("No files were attached.")
    return found

def save_upload(filename: str, data: bytes, content_type: str = "application/octet-stream") -> dict:
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Each file must be 20 MB or smaller.")
    clean = _safe_name(filename)
    ext = ".tar.gz" if clean.lower().endswith(".tar.gz") else Path(clean).suffix.lower()
    if ext in IMAGE_EXTENSIONS and not _image_signature(data, ext):
        raise ValueError("The file extension does not match a supported image format.")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_id = secrets.token_hex(16)
    stored_name = file_id + ext
    target = UPLOAD_DIR / stored_name
    target.write_bytes(data)
    extraction_error = None
    try:
        extracted = extract_text(target, ext)
    except Exception as exc:
        extracted = ""
        extraction_error = f"{type(exc).__name__}: {exc}"
    item = {
        "id": file_id, "name": clean, "stored_name": stored_name,
        "url": "/uploads/" + stored_name, "size": len(data),
        "content_type": mimetypes.guess_type(clean)[0] or content_type,
        "category": ("image" if ext in IMAGE_EXTENSIONS else "document" if ext in DOCUMENT_EXTENSIONS or ext in TEXT_EXTENSIONS else "media" if ext in MEDIA_EXTENSIONS else "archive" if ext in {".zip", ".tar", ".tgz", ".tar.gz"} else "file"),
        "text": extracted[:50000],
        "extraction_error": extraction_error,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    metadata = {key: value for key, value in item.items() if key != "text"}
    (UPLOAD_DIR / (file_id + ".json")).write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    return item

def list_uploads(limit: int = 100) -> list[dict]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for sidecar in sorted(UPLOAD_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:max(1, min(int(limit), 200))]:
        try:
            item = json.loads(sidecar.read_text(encoding="utf-8"))
            stored = UPLOAD_DIR / item.get("stored_name", "")
            if stored.is_file() and stored.parent.resolve() == UPLOAD_DIR.resolve():
                item["url"] = "/uploads/" + stored.name
                try:
                    item["text"] = extract_text(stored, stored.suffix.lower())[:50000]
                except Exception as exc:
                    item["text"] = ""
                    item["extraction_error"] = f"{type(exc).__name__}: {exc}"
                items.append(item)
        except (OSError, ValueError, TypeError):
            continue
    return items

class _VisibleHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0
    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self.hidden += 1
    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style", "noscript", "svg"} and self.hidden:
            self.hidden -= 1
    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def _xml_text(data: bytes) -> str:
    root = ElementTree.fromstring(data)
    return "\n".join(t.text or "" for t in root.iter() if t.text and t.text.strip())


def _extract_office_zip(path: Path, ext: str) -> str:
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        output = []
        if ext == ".odt" and "content.xml" in names:
            output.append(_xml_text(z.read("content.xml")))
        elif ext == ".odp":
            slides = sorted(n for n in names if re.fullmatch(r"content.xml", n))
            output.extend(_xml_text(z.read(name)) for name in slides)
        elif ext == ".ods" and "content.xml" in names:
            output.append(_xml_text(z.read("content.xml")))
        elif ext == ".epub":
            chapters = sorted(n for n in names if n.lower().endswith((".xhtml", ".html", ".htm")))
            for name in chapters[:200]:
                parser = _VisibleHTML()
                parser.feed(z.read(name).decode("utf-8", errors="replace"))
                output.append("\n".join(parser.parts))
        return "\n\n".join(part for part in output if part)[:50000]


def extract_text(path: Path, ext: str | None = None) -> str:
    ext = ext or path.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        raw = path.read_text(encoding="utf-8", errors="replace")[:50000]
        if ext in {".html", ".htm"}:
            parser = _VisibleHTML()
            parser.feed(raw)
            return "\n".join(parser.parts)[:50000]
        if ext == ".rtf":
            return re.sub(r"\\'[0-9a-fA-F]{2}|\\[a-zA-Z]+-?\d* ?|[{}]", " ", raw)[:50000]
        return raw
    if ext == ".docx":
        with zipfile.ZipFile(path) as z:
            root = ElementTree.fromstring(z.read("word/document.xml"))
        return "\n".join(t.text or "" for t in root.iter() if t.tag.endswith("}t"))[:50000]
    if ext == ".xlsx":
        with zipfile.ZipFile(path) as z:
            strings = []
            if "xl/sharedStrings.xml" in z.namelist():
                root = ElementTree.fromstring(z.read("xl/sharedStrings.xml"))
                strings = ["".join(t.text or "" for t in item.iter() if t.tag.endswith("}t")) for item in root]
            rows = []
            for name in sorted(n for n in z.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)):
                root = ElementTree.fromstring(z.read(name))
                for row in root.iter():
                    if row.tag.endswith("}row"):
                        values = []
                        for cell in row:
                            if cell.tag.endswith("}c"):
                                value = next((v.text for v in cell if v.tag.endswith("}v")), "")
                                if cell.attrib.get("t") == "s" and value:
                                    value = strings[int(value)]
                                elif cell.attrib.get("t") == "inlineStr":
                                    value = "".join(t.text or "" for t in cell.iter() if t.tag.endswith("}t"))
                                values.append(value or "")
                        rows.append("\t".join(values))
            return "\n".join(rows)[:50000]
    if ext == ".pptx":
        with zipfile.ZipFile(path) as z:
            slides = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)), key=lambda n: int(re.search(r"slide(\d+)", n).group(1)))
            output = []
            for name in slides:
                root = ElementTree.fromstring(z.read(name))
                output.append(" ".join(t.text or "" for t in root.iter() if t.tag.endswith("}t")))
            return "\n\n".join(output)[:50000]
    if ext in {".odt", ".ods", ".odp", ".epub"}:
        return _extract_office_zip(path, ext)
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)[:50000]
        except ImportError:
            return "[PDF stored. Install requirements-media.txt to extract searchable text.]"
    return ""


def search_web_images(query: str, count: int = 12, safesearch: str = "off") -> list[dict]:
    query = str(query or "").strip()
    if not query or len(query) > 400:
        raise ValueError("Enter an image-search query up to 400 characters.")
    if safesearch not in {"off", "strict"}:
        raise ValueError("safesearch must be 'off' or 'strict'.")
    token = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
    if not token:
        raise RuntimeError("Internet image search is not configured. Set BRAVE_SEARCH_API_KEY to enable Brave image search.")
    params = urllib.parse.urlencode({"q": query, "count": max(1, min(int(count), 20)), "safesearch": safesearch})
    request = urllib.request.Request(
        "https://api.search.brave.com/res/v1/images/search?" + params,
        headers={"Accept": "application/json", "X-Subscription-Token": token, "User-Agent": "OrbitLocal/0.2"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read(5_000_000).decode("utf-8"))
    results = []
    may_include_mature = bool(payload.get("extra", {}).get("might_be_offensive", False))
    for item in payload.get("results", []):
        props = item.get("properties") or {}
        thumb = item.get("thumbnail") or {}
        source = item.get("meta_url") or {}
        image_url = props.get("url") or item.get("url")
        thumb_url = thumb.get("src")
        if not image_url or not thumb_url:
            continue
        results.append({
            "title": str(item.get("title") or "Image result")[:240],
            "image_url": image_url,
            "thumbnail_url": thumb_url,
            "page_url": item.get("url") or image_url,
            "source": item.get("source") or source.get("hostname") or "",
            "width": props.get("width"),
            "height": props.get("height"),
            "may_include_mature": may_include_mature,
        })
    return results


def get_upload(file_id: str) -> dict:
    if not isinstance(file_id, str) or len(file_id) != 32 or not all(c in "0123456789abcdef" for c in file_id.lower()):
        raise ValueError("Invalid upload identifier.")
    sidecar = UPLOAD_DIR / (file_id + ".json")
    if not sidecar.is_file() or sidecar.resolve().parent != UPLOAD_DIR.resolve():
        raise FileNotFoundError("Uploaded file not found.")
    item = json.loads(sidecar.read_text(encoding="utf-8"))
    target = (UPLOAD_DIR / item.get("stored_name", "")).resolve()
    if UPLOAD_DIR.resolve() not in target.parents or not target.is_file():
        raise FileNotFoundError("Uploaded file data is missing.")
    item["url"] = "/uploads/" + target.name
    item["text"] = extract_text(target, target.suffix.lower())[:50000]
    return item


def delete_upload(file_id: str) -> bool:
    """Permanently remove one uploaded file and its metadata sidecar."""
    if not isinstance(file_id, str) or len(file_id) != 32 or not all(c in "0123456789abcdef" for c in file_id.lower()):
        raise ValueError("Invalid upload identifier.")
    root = UPLOAD_DIR.resolve()
    sidecar = (UPLOAD_DIR / (file_id + ".json")).resolve()
    if root not in sidecar.parents or not sidecar.is_file():
        raise FileNotFoundError("Uploaded file not found.")
    item = json.loads(sidecar.read_text(encoding="utf-8"))
    stored = (UPLOAD_DIR / str(item.get("stored_name", ""))).resolve()
    if root not in stored.parents:
        raise ValueError("Upload path escaped the media directory.")
    stored.unlink(missing_ok=True)
    sidecar.unlink(missing_ok=True)
    return True
