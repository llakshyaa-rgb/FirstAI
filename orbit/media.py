"""Orbit media intake, safe local storage, text extraction and web image search.

No uploaded file is executed. Image search uses Brave Search when a key is configured.
"""
from __future__ import annotations
import base64
import hashlib
import json
import mimetypes
import os
import re
import secrets
import urllib.parse
import urllib.request
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".jsonl", ".xml", ".html", ".htm", ".py", ".js", ".ts", ".tsx", ".jsx", ".css", ".sql", ".yaml", ".yml", ".toml", ".ini", ".log"}
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx"}
MEDIA_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".mov", ".webm", ".mkv"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | TEXT_EXTENSIONS | DOCUMENT_EXTENSIONS | MEDIA_EXTENSIONS | {".zip"}

def _safe_name(name: str) -> str:
    name = Path(str(name or "upload")).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return (name or "upload")[:120]

def _image_signature(data: bytes, ext: str) -> bool:
    if ext in {".jpg", ".jpeg"}: return data.startswith(b"\xff\xd8\xff")
    if ext == ".png": return data.startswith(b"\x89PNG\r\n\x1a\n")
    if ext == ".gif": return data.startswith((b"GIF87a", b"GIF89a"))
    if ext == ".webp": return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return True

def parse_multipart(content_type: str, body: bytes):
    if len(body) > MAX_UPLOAD_BYTES + 1024 * 1024:
        raise ValueError("Upload is larger than the 20 MB limit.")
    envelope = (f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n").encode("ascii") + body
    message = BytesParser(policy=policy.default).parsebytes(envelope)
    if not message.is_multipart():
        raise ValueError("Expected a multipart file upload.")
    found = []
    for part in message.iter_parts():
        filename = part.get_filename()
        if not filename:
            continue
        data = part.get_payload(decode=True) or b""
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
    ext = Path(clean).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext or '(no extension)'}.")
    if ext in IMAGE_EXTENSIONS and not _image_signature(data, ext):
        raise ValueError("The file extension does not match a supported image format.")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_id = secrets.token_hex(16)
    stored_name = file_id + ext
    target = UPLOAD_DIR / stored_name
    target.write_bytes(data)
    extracted = extract_text(target, ext)
    return {
        "id": file_id, "name": clean, "stored_name": stored_name,
        "url": "/uploads/" + stored_name, "size": len(data),
        "content_type": mimetypes.guess_type(clean)[0] or content_type,
        "category": ("image" if ext in IMAGE_EXTENSIONS else "document" if ext in DOCUMENT_EXTENSIONS or ext in TEXT_EXTENSIONS else "media" if ext in MEDIA_EXTENSIONS else "archive"),
        "text": extracted[:50000],
        "sha256": hashlib.sha256(data).hexdigest(),
    }

def extract_text(path: Path, ext: str | None = None) -> str:
    ext = ext or path.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8", errors="replace")[:50000]
    if ext == ".docx":
        with zipfile.ZipFile(path) as z:
            root = ElementTree.fromstring(z.read("word/document.xml"))
        return "\n".join(t.text or "" for t in root.iter() if t.tag.endswith("}t"))
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
                                values.append(value or "")
                        rows.append("\t".join(values))
            return "\n".join(rows)
    if ext == ".pptx":
        with zipfile.ZipFile(path) as z:
            slides = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)), key=lambda n: int(re.search(r"slide(\d+)", n).group(1)))
            output = []
            for name in slides:
                root = ElementTree.fromstring(z.read(name))
                output.append(" ".join(t.text or "" for t in root.iter() if t.tag.endswith("}t")))
            return "\n\n".join(output)
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)[:50000]
        except ImportError:
            return "[PDF stored. Install pypdf to extract searchable text.]"
    return ""

def search_web_images(query: str, count: int = 12) -> list[dict]:
    query = str(query or "").strip()
    if not query or len(query) > 400:
        raise ValueError("Enter an image-search query up to 400 characters.")
    token = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
    if not token:
        raise RuntimeError("Internet image search is not configured. Set BRAVE_SEARCH_API_KEY to enable Brave image search.")
    params = urllib.parse.urlencode({"q": query, "count": max(1, min(int(count), 20)), "safesearch": "off"})
    request = urllib.request.Request(
        "https://api.search.brave.com/res/v1/images/search?" + params,
        headers={"Accept": "application/json", "X-Subscription-Token": token, "User-Agent": "OrbitLocal/0.2"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read(5_000_000).decode("utf-8"))
    results = []
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
            "page_url": item.get("url") or source.get("url") or image_url,
            "source": source.get("hostname") or "",
            "width": item.get("properties", {}).get("width"),
            "height": item.get("properties", {}).get("height"),
            "is_mature": bool(item.get("is_mature", False)),
        })
    return results
