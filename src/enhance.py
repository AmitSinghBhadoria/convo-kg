"""Speech enhancement (denoise) stage — always runs.
DeepFilterNet runs in an isolated venv (.venv-denoise, torchaudio<2.1) because it is
incompatible with the ASR stack's torchaudio 2.x; we invoke it as a subprocess and
capture its output so the main test run stays pristine.
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "work"
DENOISE_PY = ROOT / ".venv-denoise" / "bin" / "python"
WORKER = ROOT / "scripts" / "denoise_worker.py"

def run(clip: str) -> Path:
    src = ROOT / "data" / "raw" / f"{clip}.wav"
    dst = WORK / f"{clip}.clean.wav"
    WORK.mkdir(parents=True, exist_ok=True)
    if not Path(DENOISE_PY).exists():
        raise RuntimeError(
            f"denoise venv missing at {DENOISE_PY} — create it (see README 'denoise setup')")
    proc = subprocess.run([str(DENOISE_PY), str(WORKER), str(src), str(dst)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"denoise worker failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    print("wrote", dst)
    return dst

if __name__ == "__main__":
    run(sys.argv[1])
