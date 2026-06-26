import soundfile as sf
from pathlib import Path
import pytest

@pytest.mark.integration
def test_clips_are_16k_mono():
    for name in ["pms", "sample2", "dev"]:
        p = Path("data/raw") / f"{name}.wav"
        assert p.exists(), f"missing {p} — run scripts/prep_clips.py"
        info = sf.info(str(p))
        assert info.samplerate == 16000 and info.channels == 1
