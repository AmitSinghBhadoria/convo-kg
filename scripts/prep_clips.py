"""Build 16 kHz mono demo clips from the source mp3s. Idempotent."""
import subprocess, sys
from pathlib import Path

SRC_PMS = "PMS V_s Mutual Fund  Which one is better for you - The BlueFort BluePrint.mp3"
SRC_S2  = "sample2.mp3"
OUT = Path("data/raw"); OUT.mkdir(parents=True, exist_ok=True)

def cut(src, dst, ss=None, t=None):
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if ss: cmd += ["-ss", ss]
    if t:  cmd += ["-t", t]
    cmd += ["-i", src, "-ac", "1", "-ar", "16000", str(dst)]
    subprocess.run(cmd, check=True)
    print("wrote", dst)

if __name__ == "__main__":
    cut(SRC_PMS, OUT/"pms.wav")
    cut(SRC_S2,  OUT/"sample2.wav")
    cut(SRC_PMS, OUT/"dev.wav", ss="00:00:00", t="00:00:45")  # dense opening slice
