"""Minimal ffmpeg-backed helpers for assembling video: stills to motion,
overlays, audio muxing, and crossfade transitions. Companion to
`orbit.music` for the audio side.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------

def _ffmpeg() -> str:
    return "ffmpeg"


def _ffprobe() -> str:
    return "ffprobe"


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _size(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{1,5}x\d{1,5}", value):
        raise ValueError("size must use WIDTHxHEIGHT, for example 1920x1080")
    width, height = map(int, value.split("x"))
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    return f"{width}x{height}"


def _destination(dst: str | Path) -> str:
    path = Path(dst)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _run(args: Sequence[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [str(arg) for arg in args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        command = " ".join(str(arg) for arg in args)
        raise RuntimeError(
            f"command failed ({proc.returncode}): {command}\n"
            f"stderr:\n{proc.stderr.strip()}"
        )
    return proc


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def probe_duration(path: str | Path) -> float:
    proc = _run([
        _ffprobe(), "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        str(path),
    ])
    try:
        duration = float(proc.stdout.strip())
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Could not read duration for {path}") from exc
    if not math.isfinite(duration) or duration < 0:
        raise RuntimeError(f"Invalid media duration for {path}: {duration}")
    return duration


def probe_streams(path: str | Path) -> dict:
    proc = _run([
        _ffprobe(), "-v", "error",
        "-show_streams", "-show_format",
        "-of", "json", str(path),
    ])
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid ffprobe JSON for {path}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected ffprobe response for {path}")
    return data


def _video_stream(path: str | Path) -> dict:
    streams = probe_streams(path).get("streams", [])
    for stream in streams:
        if stream.get("codec_type") == "video":
            return stream
    raise ValueError(f"No video stream found in {path}")


# ---------------------------------------------------------------------------
# Stills -> video
# ---------------------------------------------------------------------------

def images_to_video(
    pattern: str | Path,
    dst: str | Path,
    fps: float = 30.0,
    duration: Optional[float] = None,
    size: Optional[str] = None,
    codec: str = "libx264",
    crf: int = 20,
) -> str:
    """Encode an image sequence such as 'frames/%05d.png' to video."""
    fps = _positive(fps, "fps")
    if duration is not None:
        duration = _positive(duration, "duration")
    if not isinstance(codec, str) or not codec.strip():
        raise ValueError("codec must be a non-empty string")
    if not isinstance(crf, int) or not 0 <= crf <= 51:
        raise ValueError("crf must be an integer between 0 and 51")
    if size is not None:
        size = _size(size)

    dst = _destination(dst)
    args = [
        _ffmpeg(), "-hide_banner", "-nostats", "-y",
        "-framerate", str(fps),
        "-i", str(pattern),
    ]
    if size:
        args += [
            "-vf",
            f"scale={size}:force_original_aspect_ratio=decrease,"
            f"pad={size}:(ow-iw)/2:(oh-ih)/2",
        ]
    if duration is not None:
        args += ["-t", str(duration)]
    args += [
        "-c:v", codec, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        dst,
    ]
    _run(args)
    return dst


def still_to_video(
    image: str | Path,
    dst: str | Path,
    duration: float,
    fps: float = 30.0,
    size: Optional[str] = None,
) -> str:
    """Hold a single still for `duration` seconds as a video."""
    duration = _positive(duration, "duration")
    fps = _positive(fps, "fps")
    if size is not None:
        size = _size(size)

    dst = _destination(dst)
    args = [
        _ffmpeg(), "-hide_banner", "-nostats", "-y",
        "-loop", "1", "-t", str(duration),
        "-i", str(image),
    ]
    if size:
        args += [
            "-vf",
            f"scale={size}:force_original_aspect_ratio=decrease,"
            f"pad={size}:(ow-iw)/2:(oh-ih)/2",
        ]
    args += [
        "-r", str(fps),
        "-c:v", "libx264", "-crf", "20",
        "-pix_fmt", "yuv420p",
        dst,
    ]
    _run(args)
    return dst


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def overlay(
    base: str | Path,
    top: str | Path,
    dst: str | Path,
    position: str = "10:10",
    opacity: float = 1.0,
) -> str:
    """Composite `top` over `base` at a supported position."""
    opacity = float(opacity)
    if not math.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be between 0.0 and 1.0")

    positions = {
        "center": "(W-w)/2:(H-h)/2",
        "top-left": "10:10",
        "top-right": "W-w-10:10",
        "bottom-left": "10:H-h-10",
        "bottom-right": "W-w-10:H-h-10",
    }
    if position in positions:
        pos = positions[position]
    elif isinstance(position, str) and re.fullmatch(r"\d+:\d+", position):
        pos = position
    else:
        raise ValueError(
            "position must be a supported keyword or an integer x:y pair"
        )

    dst = _destination(dst)
    filter_graph = (
        f"[1:v]format=rgba,colorchannelmixer=aa={opacity:.4f}[fg];"
        f"[0:v][fg]overlay={pos}:shortest=1[v]"
    )
    _run([
        _ffmpeg(), "-hide_banner", "-nostats", "-y",
        "-i", str(base), "-i", str(top),
        "-filter_complex", filter_graph,
        "-map", "[v]",
        "-c:v", "libx264", "-crf", "20",
        "-pix_fmt", "yuv420p",
        dst,
    ])
    return dst


def crossfade(
    a: str | Path,
    b: str | Path,
    dst: str | Path,
    duration: float = 1.0,
) -> str:
    """Crossfade from clip `a` into clip `b` over `duration` seconds."""
    duration = _positive(duration, "duration")
    stream_a = _video_stream(a)
    _video_stream(b)

    width = stream_a.get("width")
    height = stream_a.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("Could not determine dimensions of the first clip")

    duration_a = probe_duration(a)
    duration_b = probe_duration(b)
    if duration >= duration_a or duration >= duration_b:
        raise ValueError("crossfade duration must be shorter than both clips")

    offset = duration_a - duration
    dst = _destination(dst)

    filter_graph = (
        f"[0:v]settb=AVTB,setpts=PTS-STARTPTS,"
        f"scale={width}:{height},setsar=1,fps=30,format=yuv420p[v0];"
        f"[1:v]settb=AVTB,setpts=PTS-STARTPTS,"
        f"scale={width}:{height},setsar=1,fps=30,format=yuv420p[v1];"
        f"[v0][v1]xfade=transition=fade:"
        f"duration={duration:.6f}:offset={offset:.6f}[v]"
    )
    _run([
        _ffmpeg(), "-hide_banner", "-nostats", "-y",
        "-i", str(a), "-i", str(b),
        "-filter_complex", filter_graph,
        "-map", "[v]",
        "-an",
        "-c:v", "libx264", "-crf", "20",
        "-pix_fmt", "yuv420p",
        dst,
    ])
    return dst


def _concat_quote(path: Path) -> str:
    """Escape a path for the FFmpeg concat demuxer list format."""
    value = str(path.resolve())
    return "'" + value.replace("\\", "\\\\").replace("'", "'\\''") + "'"


def concat(clips: Sequence[str | Path], dst: str | Path) -> str:
    """Concatenate clips with matching codec, resolution, and stream layout."""
    if not clips:
        raise ValueError("clips must contain at least one input")

    dst = _destination(dst)
    list_path: Optional[Path] = None

    try:
        parent = Path(dst).parent
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".concat.txt",
            prefix="orbit-",
            dir=str(parent),
            delete=False,
        ) as stream:
            list_path = Path(stream.name)
            for clip in clips:
                stream.write(f"file {_concat_quote(Path(clip))}\n")

        _run([
            _ffmpeg(), "-hide_banner", "-nostats", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            dst,
        ])
    finally:
        if list_path is not None:
            list_path.unlink(missing_ok=True)

    return dst


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def mux_audio(
    video: str | Path,
    audio: str | Path,
    dst: str | Path,
    shortest: bool = True,
    codec: str = "aac",
) -> str:
    """Attach an audio track to a video, re-encoding audio only."""
    if not isinstance(codec, str) or not codec.strip():
        raise ValueError("codec must be a non-empty string")

    dst = _destination(dst)
    args = [
        _ffmpeg(), "-hide_banner", "-nostats", "-y",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", codec,
    ]
    if shortest:
        args.append("-shortest")
    args.append(dst)
    _run(args)
    return dst


def extract_audio(video: str | Path, dst: str | Path) -> str:
    """Pull the audio track out of a video to a standalone WAV file."""
    dst = _destination(dst)
    _run([
        _ffmpeg(), "-hide_banner", "-nostats", "-y",
        "-i", str(video),
        "-map", "0:a:0",
        "-vn", "-c:a", "pcm_s16le",
        dst,
    ])
    return dst


# ---------------------------------------------------------------------------
# Full assembly
# ---------------------------------------------------------------------------

def build_clip(
    image: str | Path,
    audio: str | Path,
    dst: str | Path,
    size: str = "1920x1080",
    fps: float = 30.0,
) -> str:
    """Still + soundtrack -> finished clip trimmed to the audio length."""
    fps = _positive(fps, "fps")
    size = _size(size)
    duration = probe_duration(audio)
    if duration <= 0:
        raise ValueError("audio must have a positive duration")

    dst = _destination(dst)
    destination = Path(dst)
    tmp = destination.with_name(
        f"{destination.stem}.videoonly-{uuid.uuid4().hex}.mp4"
    )

    try:
        still_to_video(image, tmp, duration, fps=fps, size=size)
        mux_audio(tmp, audio, dst, shortest=True)
    finally:
        tmp.unlink(missing_ok=True)

    return dst
