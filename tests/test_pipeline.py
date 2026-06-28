import src.pipeline as P


class _FS:  # minimal FactSet stand-in
    def __init__(self, facts): self.facts = facts


class _F:
    def __init__(self, s): self.statement = s


def _patch(monkeypatch, *, fail=None):
    calls = []

    def enh(clip): calls.append("enh")

    def dia(clip):
        calls.append("dia")
        # write a 2-utterance transcript the orchestrator will read
        from src.contracts import Transcript, Utterance
        import json, pathlib
        from src.config import load_config
        w = pathlib.Path(load_config().paths.work); w.mkdir(parents=True, exist_ok=True)
        tr = Transcript(clip=clip, utterances=[
            Utterance(speaker="SPEAKER_00", text="hello", start=0.0, end=1.0),
            Utterance(speaker="SPEAKER_01", text="world", start=1.0, end=2.0)])
        (w / f"{clip}.transcript.json").write_text(tr.model_dump_json())

    def ext(clip, cfg=None, llm=None): calls.append("ext"); return _FS([_F("fact one"), _F("fact two")])

    if fail == "enh":
        def enh(clip): calls.append("enh"); raise RuntimeError("denoise boom")
    elif fail == "dia":
        def dia(clip): calls.append("dia"); raise RuntimeError("pyannote boom")
    elif fail == "dia_nofile":
        def dia(clip):
            calls.append("dia")
            # deliberately write no transcript file; remove leftovers from other runs
            import pathlib
            from src.config import load_config
            p = pathlib.Path(load_config().paths.work) / f"{clip}.transcript.json"
            p.unlink(missing_ok=True)
    elif fail == "ext":
        def ext(clip, cfg=None, llm=None): calls.append("ext"); raise RuntimeError("extract boom")

    monkeypatch.setattr(P, "enhance_run", enh)
    monkeypatch.setattr(P, "diarize_asr_run", dia)
    monkeypatch.setattr(P, "extract", ext)
    return calls


def test_run_live_sequences_and_emits(monkeypatch):
    _patch(monkeypatch)
    evs = list(P.run_live("uploadX"))
    kinds = [e["event"] for e in evs]
    assert kinds[0] == "stage" and kinds[-1] == "done"
    assert kinds.count("transcript_line") == 2
    assert [e["data"]["text"] for e in evs if e["event"] == "fact"] == ["fact one", "fact two"]
    # extraction stage announced before facts; done last
    assert "fact" in kinds and kinds.index("fact") < kinds.index("done")


def test_run_live_stage_failure_emits_error_and_stops(monkeypatch):
    _patch(monkeypatch, fail="dia")
    evs = list(P.run_live("uploadX"))
    assert evs[-1]["event"] == "error"
    assert evs[-1]["data"]["stage"] == "Diarization"
    assert "pyannote" in evs[-1]["data"]["message"]
    assert not any(e["event"] == "fact" for e in evs)  # stopped before extraction


def test_enhance_failure_emits_error_and_stops(monkeypatch):
    _patch(monkeypatch, fail="enh")
    evs = list(P.run_live("uploadX"))
    assert evs[-1]["event"] == "error"
    assert evs[-1]["data"]["stage"] == "Speech enhancement"
    assert "denoise" in evs[-1]["data"]["message"]
    assert not any(e["event"] in ("transcript_line", "fact") for e in evs)


def test_extract_failure_emits_error_and_stops(monkeypatch):
    _patch(monkeypatch, fail="ext")
    evs = list(P.run_live("uploadX"))
    assert evs[-1]["event"] == "error"
    assert evs[-1]["data"]["stage"] == "Fact extraction"
    assert "extract" in evs[-1]["data"]["message"]
    assert any(e["event"] == "transcript_line" for e in evs)  # transcript emitted before error
    assert not any(e["event"] in ("fact", "done") for e in evs)


def test_transcript_load_failure_emits_diarization_error(monkeypatch):
    _patch(monkeypatch, fail="dia_nofile")
    evs = list(P.run_live("uploadY"))  # unique clip to avoid stale file from other tests
    assert evs[-1]["event"] == "error"
    assert evs[-1]["data"]["stage"] == "Diarization"
    assert not any(e["event"] in ("transcript_line", "fact", "done") for e in evs)
