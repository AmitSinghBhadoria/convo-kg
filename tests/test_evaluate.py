import json
import pytest
from pathlib import Path
from src.contracts import Transcript, Utterance, CurvePoint, SpotCheckRow
from src.evaluate import snr_curve, retrieve_answer, downstream_spotcheck, emit_results, cliff_index, _y_limits


class FakeLLM:
    """Deterministic fake: bag-of-words embeddings + echo-the-quotes compose."""
    VOCAB = ["pms", "mutual", "fund", "minimum", "investment", "younger"]

    def embed(self, texts):
        return [[float(t.lower().count(w)) for w in self.VOCAB] for t in texts]

    def chat_json(self, system, user, schema):
        # Echo the user message so the test can see which quote was retrieved.
        return {"answer": user}


def _t(text):
    return Transcript(clip="c", utterances=[Utterance(speaker="A", text=text, start=0, end=1)])


def test_snr_curve_scores_each_noisy_against_baseline_preserving_order():
    base = _t("alpha beta gamma delta")
    noisy = {
        "20": _t("alpha beta gamma delta"),   # identical -> 1.0
        "0": _t("zzz qqq"),                    # very different -> low
    }
    pts = snr_curve(base, noisy)
    assert [p.snr for p in pts] == ["20", "0"]          # dict order preserved
    assert pts[0].similarity == 1.0                      # self-similarity is exactly 1
    assert pts[1].similarity < pts[0].similarity          # degraded scores strictly lower
    assert 0.0 <= pts[1].similarity <= 1.0


def test_retrieve_answer_grounds_in_nearest_utterance():
    t = Transcript(clip="x", utterances=[
        Utterance(speaker="A", text="The minimum investment for a PMS is fifty lakh", start=0, end=1),
        Utterance(speaker="B", text="We work with a lot of younger investors", start=1, end=2),
    ])
    ans = retrieve_answer("what is the minimum investment for a PMS", t, FakeLLM(), k=1)
    assert "fifty lakh" in ans          # nearest utterance retrieved + composed
    assert "younger" not in ans         # off-topic utterance not retrieved at k=1


def test_retrieve_answer_handles_empty_transcript():
    empty = Transcript(clip="x", utterances=[])
    ans = retrieve_answer("anything", empty, FakeLLM())
    assert isinstance(ans, str) and ans                 # graceful non-empty string, no crash


def test_downstream_spotcheck_zips_rows_without_verdict():
    rows = downstream_spotcheck(
        questions=["q1", "q2"],
        clean_answers=["c1", "c2"],
        degraded_answers=["d1", "d2"],
        degraded_snr=5,
    )
    assert [r.question for r in rows] == ["q1", "q2"]
    assert rows[0].clean_answer == "c1" and rows[0].degraded_answer == "d1"
    assert all(r.degraded_snr == 5 for r in rows)
    assert all(isinstance(r, SpotCheckRow) for r in rows)
    # SpotCheckRow has no verdict/score field — illustrative only
    assert "verdict" not in SpotCheckRow.model_fields and "score" not in SpotCheckRow.model_fields


def test_y_limits_are_a_fixed_honest_floor_ignoring_data():
    # Even a curve that never drops below 0.9 must still plot from 0 — no
    # auto-scaling to dramatize a small drop.
    c = [CurvePoint(snr="20", similarity=0.92), CurvePoint(snr="0", similarity=0.90)]
    assert _y_limits(c) == (0.0, 1.02)


def test_cliff_index_finds_point_after_steepest_drop():
    c = [CurvePoint(snr="20", similarity=0.95),
         CurvePoint(snr="10", similarity=0.90),
         CurvePoint(snr="5", similarity=0.50)]      # steepest drop is 0.90 -> 0.50
    assert cliff_index(c) == 2


def test_emit_results_writes_json_and_png_with_both_honesty_labels(tmp_path):
    curve = [CurvePoint(snr="20", similarity=0.95), CurvePoint(snr="0", similarity=0.40)]
    sc = [SpotCheckRow(question="q", clean_answer="a", degraded_answer="b", degraded_snr=5)]
    pj, pp = tmp_path / "r.json", tmp_path / "r.png"
    j, p = emit_results(curve, sc, {"noise": "noices/cafe_16k.wav"}, str(pj), str(pp))
    data = json.loads(pj.read_text())
    assert data["curve"][0]["snr"] == "20"
    assert data["spotcheck"][0]["question"] == "q"
    # both required honesty labels present in the machine-readable artifact
    assert "not a calibrated curve" in data["labels"]["spotcheck"]
    assert "not the full product path" in data["labels"]["spotcheck"]
    assert "not WER" in data["labels"]["curve"]
    assert pp.exists() and pp.stat().st_size > 0      # a real PNG was rendered


@pytest.mark.integration
def test_evaluate_produces_five_point_curve_over_real_sweep_artifacts():
    from src.evaluate import evaluate
    res = evaluate()
    pts = res.curve
    assert len(pts) == 5                                  # all 5 SNR points, never fewer
    assert all(0.0 <= p.similarity <= 1.0 for p in pts)
    by = {p.snr: p.similarity for p in pts}
    # LOOSE monotonicity ONLY: the clean end (20 dB) must beat the worst end
    # (0 dB) and the general trend is downward. NOT strict point-by-point — a
    # bumpy real curve (e.g. 5 dB slightly above 10 dB from ASR quirks) is valid;
    # the cliff/shape is the finding, not strict monotonicity.
    assert by["20"] >= by["0"]
    assert len(res.spotcheck) >= 2                         # golden questions answered both sides
    assert Path("data/ground_truth/snr_results.json").exists()
    assert Path("data/ground_truth/snr_curve.png").exists()
