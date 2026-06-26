"""diarize_asr stage — orchestrates the .venv-asr worker (subprocess) and validates output.
The heavy ASR/diarization runs in .venv-asr; this stays in the (torch-free) main env.
"""
import os, subprocess, sys
from pathlib import Path
from dotenv import load_dotenv
from src.contracts import Transcript

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "work"
ASR_PY = ROOT / ".venv-asr" / "bin" / "python"
WORKER = ROOT / "scripts" / "asr_worker.py"

def run(clip: str, mode: str = "dev") -> Path:
    load_dotenv(ROOT / ".env")
    if not Path(ASR_PY).exists():
        raise RuntimeError(f"audio venv missing at {ASR_PY} — build it (README: requirements-asr.txt)")
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN not set (needed for pyannote) — add it to .env")
    proc = subprocess.run([str(ASR_PY), str(WORKER), clip, mode],
                          cwd=str(ROOT), env={**os.environ}, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"asr worker failed (rc={proc.returncode}):\n{proc.stderr[-3000:]}")
    dst = WORK / f"{clip}.transcript.json"
    Transcript.model_validate_json(dst.read_text())     # validate worker output against contract
    print("wrote", dst)
    return dst

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "dev")
