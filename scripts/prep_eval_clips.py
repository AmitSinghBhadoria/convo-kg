"""Reproducible Phase 4 sweep: cut the eval slice, mix real café babble at each
SNR, run the real denoise + ASR pipeline -> six transcripts under data/work/.

Long-pole compute (~20 min on M4). Config-driven (config.yaml eval.*). The
transcripts it writes are consumed by src/evaluate.py — this script is the
documented, reproducible source of those artifacts (it replaces the scratch
scratch_sweep.py used during bring-up).

Run: source .venv/bin/activate && python -m scripts.prep_eval_clips
"""
import subprocess
import sys
from pathlib import Path

import soundfile as sf

import add_noise as an
from src.config import load_config

SR = 16000


def write_and_run(clip: str, audio, raw_dir: Path) -> None:
    sf.write(str(raw_dir / f"{clip}.wav"), audio, SR)
    print(f"[{clip}] enhance...", flush=True)
    subprocess.run([sys.executable, "-m", "src.enhance", clip], check=True)
    print(f"[{clip}] diarize_asr...", flush=True)
    subprocess.run([sys.executable, "-m", "src.diarize_asr", clip], check=True)
    print(f"[{clip}] done", flush=True)


def main() -> None:
    cfg = load_config()
    ec = cfg.eval
    raw = Path(cfg.paths.raw)
    speech = an.load_audio(str(raw / f"{ec.source_clip}.wav"), sr=SR)
    sl = speech[ec.slice_start_s * SR:ec.slice_end_s * SR]
    noise = an.load_audio(ec.noise_path, sr=SR)
    print(f"slice {len(sl) / SR:.0f}s  noise {len(noise) / SR:.0f}s", flush=True)

    write_and_run(f"{ec.clip_prefix}_clean", sl, raw)
    for snr in ec.snr_levels:
        mixed = an.mix(sl, an.fit_noise(noise, len(sl)), snr)
        write_and_run(f"{ec.clip_prefix}_snr{snr}", mixed, raw)
    print("SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
