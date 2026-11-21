"""orbit/music.py

Music bed management: probe, normalize, loop, duck, and mix audio tracks
against video. All operations are thin ffmpeg/ffprobe wrappers that return
paths to intermediate files and never mutate inputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _ffmpeg() -> str:
    return "ffmpeg"


def _ffprobe() -> str:
    return "ffprobe"


def _run(args: List[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '..join(args)}\n"
            f"{proc.stderr.decode(errors='replace')}"
        )
    return proc


def _probe_json(path: str | Path) -> Dict:
    proc = _run([
        _ffprobe(), "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ])
    return json.loads(proc.stdout.decode())


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def probe_duration(path: str | Path) -> float:
    """Duration in seconds. Reads format.duration; falls back to stream."""
    info = _probe_json(path)
    fmt = info.get("format", {})
    dur = fmt.get("duration")
    if dur is not None:
        return float(dur)
    for stream in info.get("streams", []):
        if "duration" in stream:
            return float(stream["duration"])
    raise RuntimeError(f"could not determine duration for {path}")


def probe_audio_streams(path: str | Path) -> List[Dict]:
    """Return only audio stream descriptors."""
    info = _probe_json(path)
    return [
        s for s in info.get("streams", [])
        if s.get("codec_type") == "audio"
    ]


def has_audio(path: str | Path) -> bool:
    return bool(probe_audio_streams(path))


def probe_loudness(path: str | Path) -> Dict[str, float]:
    """Measure integrated loudness via ebur128. Returns LUFS + true peak."""
    proc = _run([
        _ffmpeg(), "-hide_banner", "-nostats",
        "-i", str(path),
        "-af", "ebur128=peak=true",
        "-f", "null", "-",
    ])

    # ebur128 writes its summary to stderr. Parse the final block.
    text = proc.stderr.decode(errors="replace")
    summary = text.rsplit("Summary:", 1)
    if len(summary) != 2:
        raise RuntimeError(f"ebur128 produced no summary for {path}")
    block = summary[1]

    result: Dict[str, float] = {}
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("I:") and "LUFS" in line:
            result["integrated_lufs"] = float(line.split()[1])
        elif line.startswith("LRA:") and "LU" in line:
            result["lra_lu"] = float(line.split()[1])
        elif line.startswith("Peak:") and "dBFS" in line:
            result["true_peak_dbfs"] = float(line.split()[1])

    if "integrated_lufs" not in result:
        raise RuntimeError(
            f"ebur128 summary malformed for {path}:\n{block}"
        )
    return result


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(
    src: str | Path,
    dst: str | Path,
    target_lufs: float = -14.0,
    true_peak_db: float = -1.0,
    lra: float = 11.0,
) -> str:
    """Two-pass loudnorm to target_lufs integrated, ceiling at true_peak_db.

    Two-pass is used because single-pass loudnorm is an approximation; the
    second pass consumes the measured values and produces a stable result.
    """
    src, dst = str(src), str(dst)

    # Pass 1: measure.
    proc = _run([
        _ffmpeg(), "-hide_banner", "-nostats",
        "-i", src,
        "-af",
        f"loudnorm=I={target_lufs}:TP={true_peak_db}:LRA={lra}:print_format=json",
        "-f", "null", "-",
    ])
    text = proc.stderr.decode(errors="replace")

    # The JSON block is the last {...} object printed to stderr.
    start = text.rfind("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(
            f"loudnorm pass 1 emitted no JSON for {src}:\n{text}"
        )
    measured = json.loads(text[start:end + 1])

    # Pass 2: apply measured values.
    _run([
        _ffmpeg(), "-hide_banner", "-nostats", "-y",
        "-i", src,
        "-af", (
            f"loudnorm=I={target_lufs}:TP={true_peak_db}:LRA={lra}"
            f":measured_I={measured['input_i']}"
            f":measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}"
            f":measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}"
            ":linear=true"
        ),
        "-ar", "48000",
        "-c:a", "pcm_s16le",
        dst,
    ])
    return dst


# ---------------------------------------------------------------------------
# Looping / trimming
# ---------------------------------------------------------------------------

def loop_to_duration(
    src: str | Path,
    dst: str | Path,
    duration: float,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
) -> str:
    """Repeat src until it covers duration, then trim.

    Optional fades are applied after the trim so they land on the final
    cut points rather than on an arbitrary loop boundary.
    """
    src, dst = str(src), str(dst)
    filters = []

    if fade_in > 0.0:
        filters.append(f"afade=t=in:st=0:d={fade_in}")

    if fade_out > 0.0:
        fade_out_start = max(0.0, duration - fade_out)
        filters.append(
            f"afade=t=out:st={fade_out_start}:d={fade_out}"
        )

    args = [
        _ffmpeg(), "-hide_banner", "-nostats", "-y",
        "-stream_loop", "-1",
        "-i", src,
        "-t", f"{duration:.3f}",
    ]

    if filters:
        args += ["-af", ",".join(filters)]

    args += ["-ar", "48000", "-c:a", "pcm_s16le", dst]
    _run(args)
    return dst


def trim(
    src: str | Path,
    dst: str | Path,
    start: float = 0.0,
    duration: Optional[float] = None,
) -> str:
    """Cut a segment out of src. start is in seconds from the head."""
    src, dst = str(src), str(dst)
    args = [
        _ffmpeg(), "-hide_banner", "-nostats", "-y",
        "-ss", f"{start:.3f}",
        "-i", src,
    ]

    if duration is not None:
        args += ["-t", f"{duration:.3f}"]

    args += ["-ar", "48000", "-c:a", "pcm_s16le", dst]
    _run(args)
    return dst


# ---------------------------------------------------------------------------
# Ducking
# ---------------------------------------------------------------------------

def duck(
    music: str | Path,
    voice: str | Path,
    dst: str | Path,
    threshold: float = 0.05,
    ratio: float = 8.0,
    attack: float = 20.0,
    release: float = 400.0,
    makeup: float = 1.0,
) -> str:
    """Sidechain-compress music against voice, writing the ducked music.

    Voice is only used as the sidechain input; it is not mixed here.
    The result is a music-only track that dips under dialogue.
    Mix it with the voice afterwards via mix().
    """
    music, voice, dst = str(music), str(voice), str(dst)

    _run([
        _ffmpeg(), "-hide_banner", "-nostats", "-y",
        "-i", music,
        "-i", voice,
        "-filter_complex", (
            f"[0:a][1:a]sidechaincompress="
            f"threshold={threshold}:ratio={ratio}"
            f":attack={attack}:release={release}"
            f":makeup={makeup}"
            "[ducked]"
        ),
        "-map", "[ducked]",
        "-ar", "48000", "-c:a", "pcm_s16le", dst,
    ])
    return dst


# ---------------------------------------------------------------------------
# Mixing
# ---------------------------------------------------------------------------

def mix(
    tracks: List[str | Path],
    dst: str | Path,
    weights: Optional[List[float]] = None,
    duration: Optional[float] = None,
) -> str:
    """Sum tracks into a single stereo output.

    Weights are per-track linear gains (default 1.0 each). If duration
    is given, output is trimmed to that many seconds; otherwise it runs
    to the longest input. Inputs are normalized to 48 kHz stereo first
    so the amix node sees uniform streams.
    """
    if not tracks:
        raise ValueError("mix() requires at least one track")

    if weights is None:
        weights = [1.0] * len(tracks)

    if len(weights) != len(tracks):
        raise ValueError("weights length must match tracks length")

    dst = str(dst)
    args = [_ffmpeg(), "-hide_banner", "-nostats", "-y"]

    for track in tracks:
        args += ["-i", str(track)]

    chains = []
    labels = []

    for i, weight in enumerate(weights):
        chains.append(
            f"[{i}:a]aresample=48000,"
            f"aformat=channel_layouts=stereo,"
            f"volume={weight}[a{i}]"
        )
        labels.append(f"[a{i}]")

    filter_complex = (
        ";".join(chains)
        + ";"
        + "".join(labels)
        + f"amix=inputs={len(tracks)}:normalize=0[out]"
    )

    args += ["-filter_complex", filter_complex, "-map", "[out]"]

    if duration is not None:
        args += ["-t", f"{duration:.3f}"]

    args += ["-ar", "48000", "-c:a", "pcm_s16le", dst]
    _run(args)
    return dst


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------

def prepare_bed(
    src: str | Path,
    workdir: str | Path,
    duration: float,
    target_lufs: float = -18.0,
    fade_in: float = 1.0,
    fade_out: float = 2.0,
) -> str:
    """Normalize, loop to duration, and fade a music bed in one call.

    Music beds sit under dialogue, so the default target is -18 LUFS
    rather than the -14 used for foreground content.
    Intermediate files live in workdir; the returned path is the finished bed.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem

    normalized = workdir / f"{stem}.norm.wav"
    normalize(src, normalized, target_lufs=target_lufs)

    bed = workdir / f"{stem}.bed.wav"
    loop_to_duration(
        normalized, bed, duration,
        fade_in=fade_in,
        fade_out=fade_out,
    )
    return str(bed)


def build_soundtrack(
    music: str | Path,
    voice: str | Path,
    workdir: str | Path,
    duration: float,
    music_lufs: float = -18.0,
    voice_lufs: float = -16.0,
    master_lufs: float = -14.0,
    duck_threshold: float = 0.05,
    duck_ratio: float = 8.0,
    fade_in: float = 1.0,
    fade_out: float = 2.0,
) -> str:
    """Build a normalized voice + looped music soundtrack.

    Pipeline:
    normalize voice, prepare music bed, duck music against voice, mix,
    then master the combined output to master_lufs.

    Intermediates are retained in workdir for inspection or reruns.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    v_stem = Path(voice).stem
    m_stem = Path(music).stem

    # 1. Normalize dialogue to its target level.
    voice_norm = workdir / f"{v_stem}.norm.wav"
    normalize(voice, voice_norm, target_lufs=voice_lufs)

    # 2. Prepare the music bed: normalize, loop, and fade.
    bed = prepare_bed(
        music, workdir, duration,
        target_lufs=music_lufs,
        fade_in=fade_in,
        fade_out=fade_out,
    )

    # 3. Duck the bed against the normalized voice.
    ducked = workdir / f"{m_stem}.ducked.wav"
    duck(
        bed, voice_norm, ducked,
        threshold=duck_threshold,
        ratio=duck_ratio,
    )

    # 4. Mix voice and ducked bed.
    mixed = workdir / "mix.wav"
    mix([voice_norm, ducked], mixed, duration=duration)

    # 5. Master the final mix to its own target.
    final = workdir / "soundtrack.wav"
    normalize(mixed, final, target_lufs=master_lufs)

    return str(final)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audiomix",
        description="Normalize, loop, duck, and mix audio via ffmpeg.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    n = sub.add_parser(
        "normalize",
        help="Two-pass EBU R128 loudness normalization",
    )
    n.add_argument("src")
    n.add_argument("dst")
    n.add_argument("--lufs", type=float, default=-16.0)
    n.add_argument("--tp", type=float, default=-1.5)
    n.add_argument("--lra", type=float, default=11.0)

    lp = sub.add_parser("loop", help="Loop a file to a target duration")
    lp.add_argument("src")
    lp.add_argument("dst")
    lp.add_argument("--duration", type=float, required=True)
    lp.add_argument("--fade-in", type=float, default=0.0)
    lp.add_argument("--fade-out", type=float, default=0.0)

    tr = sub.add_parser("trim", help="Cut a segment")
    tr.add_argument("src")
    tr.add_argument("dst")
    tr.add_argument("--start", type=float, default=0.0)
    tr.add_argument("--duration", type=float, default=None)

    d = sub.add_parser("duck", help="Sidechain-duck music against voice")
    d.add_argument("music")
    d.add_argument("voice")
    d.add_argument("dst")
    d.add_argument("--threshold", type=float, default=0.05)
    d.add_argument("--ratio", type=float, default=8.0)

    mx = sub.add_parser("mix", help="Sum multiple tracks")
    mx.add_argument("tracks", nargs="+")
    mx.add_argument("dst")
    mx.add_argument("--weights", type=float, nargs="+", default=None)
    mx.add_argument("--duration", type=float, default=None)

    sb = sub.add_parser(
        "soundtrack",
        help="Normalize + bed + duck + mix pipeline",
    )
    sb.add_argument("music")
    sb.add_argument("voice")
    sb.add_argument("workdir")
    sb.add_argument("--duration", type=float, required=True)
    sb.add_argument("--music-lufs", type=float, default=-18.0)
    sb.add_argument("--voice-lufs", type=float, default=-16.0)
    sb.add_argument("--master-lufs", type=float, default=-14.0)
    sb.add_argument("--fade-in", type=float, default=1.0)
    sb.add_argument("--fade-out", type=float, default=2.0)
    sb.add_argument("--duck-threshold", type=float, default=0.05)
    sb.add_argument("--duck-ratio", type=float, default=8.0)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "normalize":
        out = normalize(
            args.src, args.dst,
            target_lufs=args.lufs,
            true_peak_db=args.tp,
            lra=args.lra,
        )
        print(out)
        return 0

    if args.cmd == "loop":
        out = loop_to_duration(
            args.src, args.dst, args.duration,
            fade_in=args.fade_in,
            fade_out=args.fade_out,
        )
        print(out)
        return 0

    if args.cmd == "trim":
        out = trim(
            args.src, args.dst,
            start=args.start,
            duration=args.duration,
        )
        print(out)
        return 0

    if args.cmd == "duck":
        out = duck(
            args.music, args.voice, args.dst,
            threshold=args.threshold,
            ratio=args.ratio,
        )
        print(out)
        return 0

    if args.cmd == "mix":
        out = mix(
            args.tracks, args.dst,
            weights=args.weights,
            duration=args.duration,
        )
        print(out)
        return 0

    if args.cmd == "soundtrack":
        out = build_soundtrack(
            args.music, args.voice, args.workdir,
            duration=args.duration,
            music_lufs=args.music_lufs,
            voice_lufs=args.voice_lufs,
            master_lufs=args.master_lufs,
            duck_threshold=args.duck_threshold,
            duck_ratio=args.duck_ratio,
            fade_in=args.fade_in,
            fade_out=args.fade_out,
        )
        print(out)
        return 0

    parser.error(f"unknown command: {args.cmd}")
    return 2  # unreachable; parser.error() exits


if __name__ == "__main__":
    raise SystemExit(main())
