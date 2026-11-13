"""Optional deeper processing for PDFs, archives, audio and video uploads."""
from __future__ import annotations
import os
import subprocess
import tempfile
import uuid
import zipfile
from pathlib import Path
from .media import get_upload, UPLOAD_DIR
from .vision import describe_image

_TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".jsonl", ".xml", ".html", ".htm", ".py", ".js", ".ts", ".css", ".sql", ".yaml", ".yml", ".toml", ".ini", ".log"}

def _vision_temp(data: bytes, prompt: str) -> str:
    name = uuid.uuid4().hex + ".png"
    path = UPLOAD_DIR / name
    try:
        path.write_bytes(data)
        return describe_image(name[:-4], prompt)
    finally:
        try:
            path.unlink()
        except OSError:
            pass

def _process_pdf(path: Path, extracted: str) -> str:
    if extracted.strip():
        return extracted
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError("This PDF has no extractable text. Install optional media requirements to enable scanned-page OCR.") from exc
    document = pdfium.PdfDocument(str(path))
    pages = []
    for index in range(min(len(document), 12)):
        bitmap = document[index].render(scale=1.35)
        image_bytes = bitmap.to_pil().convert("RGB")
        import io
        stream = io.BytesIO()
        image_bytes.save(stream, format="PNG")
        pages.append("Page " + str(index + 1) + ":\n" + _vision_temp(stream.getvalue(), "Read and accurately transcribe all visible text on this document page. Preserve headings, lists, numbers and table structure."))
    return "\n\n".join(pages)

def _process_zip(path: Path) -> str:
    output = []
    total = 0
    with zipfile.ZipFile(path) as archive:
        entries = [item for item in archive.infolist() if not item.is_dir()]
        if len(entries) > 200:
            raise ValueError("Archive contains too many files (limit 200).")
        for item in entries:
            if Path(item.filename).suffix.lower() not in _TEXT_SUFFIXES:
                continue
            if item.file_size > 1_000_000 or total + item.file_size > 5_000_000:
                continue
            if item.flag_bits & 1:
                continue
            data = archive.read(item)
            total += len(data)
            output.append("FILE: " + item.filename + "\n" + data.decode("utf-8", errors="replace")[:20000])
    return "\n\n".join(output)[:50000] or "Archive stored. No supported text files were found inside it."

def _transcribe_audio(path: Path) -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Audio transcription needs the optional faster-whisper package. Install requirements-media.txt.") from exc
    size = os.environ.get("ORBIT_WHISPER_MODEL", "tiny")
    device = os.environ.get("ORBIT_WHISPER_DEVICE", "cpu")
    compute = os.environ.get("ORBIT_WHISPER_COMPUTE_TYPE", "int8")
    model = WhisperModel(size, device=device, compute_type=compute)
    segments, info = model.transcribe(str(path), beam_size=1, vad_filter=True)
    lines = [segment.text.strip() for segment in segments if segment.text.strip()]
    return "Detected language: " + str(info.language) + "\n\n" + "\n".join(lines)

def _process_video(path: Path) -> str:
    ffmpeg = os.environ.get("ORBIT_FFMPEG", "ffmpeg")
    with tempfile.TemporaryDirectory(prefix="orbit-video-") as folder:
        pattern = str(Path(folder) / "frame-%02d.png")
        command = [ffmpeg, "-v", "error", "-i", str(path), "-vf", "fps=1/8,scale='min(1024,iw)':-2", "-frames:v", "6", pattern]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
        if result.returncode:
            raise RuntimeError("Video frame extraction failed: " + (result.stderr or "ffmpeg returned an error")[-1200:])
        frames = sorted(Path(folder).glob("frame-*.png"))
        if not frames:
            raise RuntimeError("No video frames could be extracted.")
        descriptions = []
        for index, frame in enumerate(frames, 1):
            descriptions.append("Frame " + str(index) + ":\n" + _vision_temp(frame.read_bytes(), "Describe the visible scene, people, objects, actions and any readable text."))
        return "\n\n".join(descriptions)

def process_upload(file_id: str, prompt: str = "Describe or analyse this attachment.") -> dict:
    item = get_upload(file_id)
    path = UPLOAD_DIR / item["stored_name"]
    ext = path.suffix.lower()
    if item["category"] == "image":
        return {"name": item["name"], "kind": "image", "content": describe_image(file_id, prompt)}
    if ext == ".pdf":
        return {"name": item["name"], "kind": "document", "content": _process_pdf(path, item.get("text", ""))}
    if ext == ".zip":
        return {"name": item["name"], "kind": "archive", "content": _process_zip(path)}
    if ext in {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".aiff", ".aif", ".opus", ".amr", ".wma"}:
        return {"name": item["name"], "kind": "audio", "content": _transcribe_audio(path)}
    if ext in {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".3gp", ".mpeg", ".mpg", ".wmv", ".flv", ".ts"}:
        return {"name": item["name"], "kind": "video", "content": _process_video(path)}
    if item.get("text"):
        return {"name": item["name"], "kind": "document", "content": item["text"]}
    return {"name": item["name"], "kind": "file", "content": "This file was safely stored, but Orbit has no content parser for its format yet."}
