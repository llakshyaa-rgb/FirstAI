"""Post-processing for generated video: trims, audio mux, resize, thumbnails."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


class VideoPostError(RuntimeError):
    pass


def _which(prog: str) -> Optional[str]:
    return shutil.which(prog)


def _run(cmd: List[str], *, timeout: int = 900) -> str:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise VideoPostError(f"timeout: {' '.join(cmd)}") from e
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        raise VideoPostError(
            f"cmd failed ({proc.returncode}): {' '.join(cmd)}\n" + "\n".join(tail)
        )
    return proc.stdout


def _ffmpeg() -> str:
    exe = _which("ffmpeg")
    if not exe:
        raise VideoPostError("ffmpeg not found on PATH")
    return exe


def _ffprobe() -> str:
    exe = _which("ffprobe")
    if not exe:
        raise VideoPostError("ffprobe not found on PATH")
    return exe


def probe(path: str | Path) -> Dict[str, Any]:
    """Return basic stream info via ffprobe."""
    out = _run(
        [
            _ffprobe(),
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    return json.loads(out or "{}")


def trim(
    src: str | Path,
    dst: str | Path,
    *,
    start: float = 0.0,
    end: Optional[float] = None,
    duration: Optional[float] = None,
) -> str:
    """Trim [start, end) or [start, start+duration). Stream-copy fast; re-encode on cut drift."""
    if end is None and duration is None:
        raise VideoPostError("trim requires end or duration")
    if end is None and duration is not None:
        end = start + duration

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _ffmpeg(), "-y",
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-i", str(src),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(dst),
        ]
    )
    return str(dst)


def trim_precise(
    src: str | Path,
    dst: str | Path,
    *,
    start: float,
    end: float,
) -> str:
    """Re-encode trim for frame-accurate cuts."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _ffmpeg(), "-y",
            "-i", str(src),
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(dst),
        ]
    )
    return str(dst)


def mux_audio(
    video: str | Path,
    audio: str | Path,
    dst: str | Path,
    *,
    shortest: bool = True,
    loop_audio: bool = False,
    audio_volume: Optional[float] = None,
) -> str:
    """Attach/replace audio track. If loop_audio, loop the audio to fill the video."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cmd = [_ffmpeg(), "-y", "-i", str(video)]
    if loop_audio:
        cmd += ["-stream_loop", "-1", "-i", str(audio)]
    else:
        cmd += ["-i", str(audio)]

    if audio_volume is not None:
        cmd += ["-filter:a", f"volume={audio_volume:.3f}"]

    cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
    if shortest:
        cmd += ["-shortest"]
    cmd += ["-movflags", "+faststart", str(dst)]

    _run(cmd)
    return str(dst)


def replace_audio(
    video: str | Path,
    audio: str | Path,
    dst: str | Path,
    **kwargs: Any,
) -> str:
    """Alias of mux_audio with loop_audio=False; strips original audio."""
    return mux_audio(video, audio, dst, loop_audio=False, **kwargs)


def resize(
    src: str | Path,
    dst: str | Path,
    *,
    width: int,
    height: int,
    keep_aspect: bool = True,
    pad_color: str = "black",
) -> str:
    """Resize to (width,height). If keep_aspect, pad instead of stretch."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if keep_aspect:
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={pad_color}"
        )
    else:
        vf = f"scale={width}:{height}"

    _run(
        [
            _ffmpeg(), "-y",
            "-i", str(src),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(dst),
        ]
    )
    return str(dst)


def change_fps(
    src: str | Path,
    dst: str | Path,
    *,
    fps: int,
    interpolate: bool = False,
) -> str:
    """Retime to target fps. interpolate=True uses minterpolate (slow, smooth)."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if interpolate:
        vf = f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1"
    else:
        vf = f"fps={fps}"

    _run(
        [
            _ffmpeg(), "-y",
            "-i", str(src),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(dst),
        ]
    )
    return str(dst)


def thumbnail(
    src: str | Path,
    dst: str | Path,
    *,
    at: float = 0.5,
    width: int = 512,
) -> str:
    """Extract a single JPEG frame at time at (seconds)."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _ffmpeg(), "-y",
            "-ss", f"{at:.3f}",
            "-i", str(src),
            "-frames:v", "1",
            "-vf", f"scale={width}:-2",
            "-q:v", "3",
            str(dst),
        ]
    )
    return str(dst)


def extract_frames(
    src: str | Path,
    out_dir: str | Path,
    *,
    fps: int = 1,
    pattern: str = "frame_%05d.png",
) -> List[str]:
    """Dump frames at fps into out_dir. Returns sorted file list."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _ffmpeg(), "-y",
            "-i", str(src),
            "-vf", f"fps={fps}",
            str(out_dir / pattern),
        ]
    )
    glob_pattern = re.sub(r"%0?\d*d", "*", pattern)
    files = sorted(str(p) for p in out_dir.glob(glob_pattern))
    return files


def concat(
    parts: List[str | Path],
    dst: str | Path,
    *,
    reencode: bool = True,
) -> str:
    """Concatenate clips in order. reencode=True handles mismatched streams."""
    if not parts:
        raise VideoPostError("concat requires at least one input")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)

    listfile = Path(dst).with_suffix(".concat.txt")
    listfile.write_text(
        "\n".join(f"file '{Path(p).resolve()}'" for p in parts) + "\n",
        encoding="utf-8",
    )

    try:
        if reencode:
            _run(
                [
                    _ffmpeg(), "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", str(listfile),
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "20",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-movflags", "+faststart",
                    str(dst),
                ]
            )
        else:
            _run(
                [
                    _ffmpeg(), "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", str(listfile),
                    "-c", "copy",
                    str(dst),
                ]
            )
    finally:
        try:
            listfile.unlink()
        except OSError:
            pass

    return str(dst)


def overlay_subtitles(
    src: str | Path,
    dst: str | Path,
    *,
    srt: str | Path,
    font_size: int = 24,
    font_color: str = "white",
    outline: int = 2,
    outline_color: str = "black",
) -> str:
    """Burn an SRT subtitle file into the video."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    style = (
        f"FontSize={font_size},PrimaryColour=&H{_bgr(font_color)},"
        f"Outline={outline},OutlineColour=&H{_bgr(outline_color)}"
    )
    vf = f"subtitles={Path(srt).resolve()}:force_style='{style}'"

    _run(
        [
            _ffmpeg(), "-y",
            "-i", str(src),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(dst),
        ]
    )
    return str(dst)


def _bgr(color: str) -> str:
    """Convert #RRGGBB or a named color to ASS BGR hex (no alpha)."""
    named = {
        "white": "FFFFFF",
        "black": "000000",
        "red": "FF0000",
        "green": "00FF00",
        "blue": "0000FF",
        "yellow": "FFFF00",
        "cyan": "00FFFF",
        "magenta": "FF00FF",
    }
    c = color.strip().lower()
    if c in named:
        rgb = named[c]
    elif c.startswith("#") and len(c) == 7:
        rgb = c[1:].upper()
    else:
        rgb = "FFFFFF"
    rr, gg, bb = rgb[0:2], rgb[2:4], rgb[4:6]
    return f"{bb}{gg}{rr}"


def finalize(
    src: str | Path,
    dst: str | Path,
    *,
    max_duration: Optional[float] = None,
    target_fps: Optional[int] = None,
    target_size: Optional[tuple[int, int]] = None,
    audio: Optional[str | Path] = None,
    thumbnail_path: Optional[str | Path] = None,
    thumbnail_at: float = 0.5,
) -> Dict[str, Any]:
    """Run the full finishing chain in one pass. Returns metadata dict."""
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    work = dst_path.with_name(f"{dst_path.stem}.{uuid.uuid4().hex}.work.mp4")
    cur = str(src)

    try:
        # 1. Optional trim to max duration.
        if max_duration is not None:
            trim(cur, work, start=0.0, duration=max_duration)
            cur = str(work)

        # 2. Resize.
        if target_size is not None:
            w, h = target_size
            resize(cur, work, width=w, height=h, keep_aspect=True)
            cur = str(work)

        # 3. Retime fps.
        if target_fps is not None:
            change_fps(cur, work, fps=target_fps, interpolate=False)
            cur = str(work)

        # 4. Attach or replace audio.
        if audio is not None:
            mux_audio(
                cur,
                audio,
                work,
                shortest=True,
                loop_audio=True,
            )
            cur = str(work)

        # 5. Final normalize pass to dst with faststart.
        if cur == str(src):
            # No transforms requested; remux streams without re-encoding.
            _run(
                [
                    _ffmpeg(), "-y",
                    "-i", cur,
                    "-c", "copy",
                    "-movflags", "+faststart",
                    str(dst),
                ]
            )
        else:
            shutil.move(cur, dst)

        # 6. Thumbnail.
        if thumbnail_path is not None:
            try:
                thumbnail(dst, thumbnail_path, at=thumbnail_at)
            except VideoPostError:
                thumbnail_path = None

        info = probe(dst)
        fmt = info.get("format", {})
        streams = info.get("streams", [])
        vstream = next(
            (s for s in streams if s.get("codec_type") == "video"), {}
        )
        astream = next(
            (s for s in streams if s.get("codec_type") == "audio"), {}
        )

        return {
            "path": str(dst),
            "duration": float(fmt.get("duration", 0.0) or 0.0),
            "size_bytes": int(fmt.get("size", 0) or 0),
            "width": int(vstream.get("width", 0) or 0),
            "height": int(vstream.get("height", 0) or 0),
            "fps": _parse_fps(vstream.get("avg_frame_rate", "0/1")),
            "video_codec": vstream.get("codec_name"),
            "has_audio": bool(astream),
            "audio_codec": astream.get("codec_name"),
            "thumbnail": str(thumbnail_path) if thumbnail_path else None,
        }
    finally:
        try:
            if work.exists():
                work.unlink()
        except OSError:
            pass


def probe_width(path: str | Path) -> int:
    info = probe(path)
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            return int(s.get("width", 0) or 0)
    return 0


def probe_height(path: str | Path) -> int:
    info = probe(path)
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            return int(s.get("height", 0) or 0)
    return 0


def _parse_fps(rate: str) -> float:
    """Parse ffprobe's 'num/den' fps string."""
    try:
        num, den = rate.split("/")
        d = float(den)
        if d == 0:
            return 0.0
        return round(float(num) / d, 3)
    except (ValueError, AttributeError):
        return 0.0
