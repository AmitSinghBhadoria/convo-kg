from src.asr_merge import merge_words_to_speakers

def test_merge_assigns_speaker_by_overlap():
    words = [{"text":"PMS","start":0.1,"end":0.4},
             {"text":"minimum","start":0.5,"end":0.9},
             {"text":"haan","start":2.1,"end":2.4}]
    turns = [("SPEAKER_00",0.0,1.0), ("SPEAKER_01",2.0,3.0)]
    utts = merge_words_to_speakers(words, turns)
    assert [u["speaker"] for u in utts] == ["SPEAKER_00","SPEAKER_01"]
    assert utts[0]["text"] == "PMS minimum" and utts[1]["text"] == "haan"


def test_run_raises_without_asr_venv(monkeypatch, tmp_path):
    import pytest, src.diarize_asr as d
    monkeypatch.setattr(d, "ASR_PY", tmp_path / "no_python")
    with pytest.raises(RuntimeError, match="audio venv"):
        d.run("dev")
