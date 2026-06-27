from src.contracts import Transcript, Utterance, CurvePoint, SpotCheckRow
from src.evaluate import snr_curve, retrieve_answer, downstream_spotcheck


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
    assert all(isinstance(p, CurvePoint) for p in pts)
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
