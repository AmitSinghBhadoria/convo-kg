from src.contracts import Transcript, Utterance, CurvePoint
from src.evaluate import snr_curve


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
