import numpy as np, soundfile as sf, pytest
from pathlib import Path

def test_run_raises_when_denoise_venv_missing(monkeypatch, tmp_path):
    import src.enhance as e
    monkeypatch.setattr(e, "DENOISE_PY", tmp_path / "no_such_python")
    with pytest.raises(RuntimeError, match="denoise venv"):
        e.run("dev")

@pytest.mark.integration
def test_enhance_produces_clean_wav():
    from src.enhance import run
    out = run("dev")
    assert Path(out).exists()
    y, sr = sf.read(str(out))
    assert sr == 16000 and y.ndim == 1 and np.isfinite(y).all()
