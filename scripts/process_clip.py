"""
scripts/process_clip.py — Trim an audio source to a fact-dense segment and
run the full offline pipeline: enhance → diarize_asr → extract.

Usage:
    python -m scripts.process_clip <clip_id> <src_path> <start_s> <dur_s>

Example:
    python -m scripts.process_clip call_100 data/raw/call_100.mp3 30 90

Outputs:
    data/raw/<clip_id>.wav          (trimmed 16 kHz mono)
    data/work/<clip_id>.clean.wav   (denoised)
    data/work/<clip_id>.transcript.json
    data/work/<clip_id>.facts.json
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def trim_to_wav(src: str, clip_id: str, start_s: float, dur_s: float) -> Path:
    """Trim <src> from <start_s> for <dur_s> seconds → data/raw/<clip_id>.wav (16 kHz mono)."""
    out = ROOT / "data" / "raw" / f"{clip_id}.wav"
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-t", str(dur_s),
        "-i", str(src),
        "-ac", "1",
        "-ar", "16000",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (rc={result.returncode}):\n{result.stderr[-2000:]}")
    print(f"[process_clip] trimmed → {out}")
    return out


def main() -> None:
    if len(sys.argv) != 5:
        print("Usage: python -m scripts.process_clip <clip_id> <src_path> <start_s> <dur_s>", file=sys.stderr)
        sys.exit(1)

    clip_id = sys.argv[1]
    src_path = sys.argv[2]
    start_s = float(sys.argv[3])
    dur_s = float(sys.argv[4])

    # Stage 0: trim source to data/raw/<clip_id>.wav
    trim_to_wav(src_path, clip_id, start_s, dur_s)

    # Stage 1: denoise
    from src.enhance import run as enhance_run
    enhance_run(clip_id)

    # Stage 2: diarize + ASR
    from src.diarize_asr import run as diarize_run
    diarize_run(clip_id)

    # Stage 3: fact extraction (no Neo4j write)
    from src.extract import extract
    factset = extract(clip_id)

    print(
        f"[process_clip] done — {len(factset.entities)} entities, "
        f"{len(factset.facts)} facts → data/work/{clip_id}.facts.json"
    )


if __name__ == "__main__":
    main()
