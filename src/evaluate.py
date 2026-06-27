"""Phase 4 controlled-SNR evaluation harness.

Pure/IO split: snr_curve + downstream_spotcheck + retrieve_answer are pure /
LLM-injected and unit-tested with a fake; emit_results does IO (JSON + PNG);
evaluate() orchestrates over the sweep transcripts already on disk.

Hero curve = transcript similarity vs SNR, oracle-free (each noisy transcript
scored against the clean slice through the identical pipeline). The downstream
spot-check is transcript-grounded retrieval (NOT the full extract->graph->Q&A
product path) — see docs/superpowers/specs/2026-06-27-atyx-convo-kg-phase4-eval-design.md.
"""
import json
from pathlib import Path

from src.contracts import CurvePoint, Provenance, SpotCheckRow, Transcript
from src.evaltools import similarity, transcript_text
from src.qa import compose_answer, top_k_statements


def snr_curve(baseline: Transcript,
              noisy: dict[str, Transcript]) -> list[CurvePoint]:
    """Score each noisy transcript against the baseline transcript.

    Oracle-free: the baseline is the clean slice through the identical pipeline.
    Order of points follows the insertion order of `noisy`.
    """
    ref = transcript_text(baseline)
    return [CurvePoint(snr=snr, similarity=similarity(transcript_text(t), ref))
            for snr, t in noisy.items()]


def retrieve_answer(question: str, transcript: Transcript, llm, k: int = 3) -> str:
    """Answer a question from ONE transcript via cosine retrieval (no graph).

    Reuses the qa semantic-fallback mechanism over the transcript's own
    utterances. This is transcript-grounded retrieval, deliberately NOT the
    extract->graph->Q&A product path — the spot-check isolates transcript->answer.
    """
    utts = transcript.utterances
    if not utts:
        return "No transcript content available to answer from."
    vecs = llm.embed([u.text for u in utts])
    stmts = [{"id": f"u{i}", "speaker": u.speaker, "text": u.text, "vec": v}
             for i, (u, v) in enumerate(zip(utts, vecs))]
    question_vec = llm.embed([question])[0]
    top = top_k_statements(question_vec, stmts, k)
    provenance = [Provenance(statement_id=s["id"], speaker=s["speaker"],
                             text=s["text"], kind="related") for s in top]
    return compose_answer(question, [], provenance, llm)


def downstream_spotcheck(questions: list[str],
                         clean_answers: list[str],
                         degraded_answers: list[str],
                         degraded_snr: int) -> list[SpotCheckRow]:
    """Assemble side-by-side clean/degraded rows. Pure; no grading verdict."""
    return [SpotCheckRow(question=q, clean_answer=ca, degraded_answer=da,
                         degraded_snr=degraded_snr)
            for q, ca, da in zip(questions, clean_answers, degraded_answers)]


CURVE_LABEL = ("transcript fidelity vs SNR — relative sequence+set similarity "
               "(not WER); reference is the clean slice through the identical pipeline")
SPOTCHECK_LABEL = ("illustrative propagation, not a calibrated curve; "
                   "transcript-grounded retrieval, not the full product path")


def cliff_index(curve: list[CurvePoint]) -> int:
    """Index of the point after the steepest consecutive similarity drop."""
    if len(curve) < 2:
        return 0
    drops = [(curve[i].similarity - curve[i + 1].similarity, i + 1)
             for i in range(len(curve) - 1)]
    return max(drops)[1]


def _y_limits(curve: list[CurvePoint]) -> tuple[float, float]:
    """Fixed honest floor at 0 — never auto-scaled to exaggerate small drops."""
    return (0.0, 1.02)


def _render_png(curve: list[CurvePoint], meta: dict, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [int(p.snr) for p in curve]
    ys = [p.similarity for p in curve]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, marker="o", color="#b8860b")
    ax.set_xlabel("SNR (dB) — café babble (higher = cleaner)")
    ax.set_ylabel("Transcript similarity vs clean baseline [0–1]")
    ax.set_ylim(*_y_limits(curve))            # honest floor
    ax.invert_xaxis()                          # easy (high SNR) left -> hard (0 dB) right
    ax.set_title("Front-end fidelity vs noise — 2-speaker PMS slice")
    if len(curve) >= 2:
        ci = cliff_index(curve)
        ax.annotate(f"cliff at {xs[ci]} dB",
                    xy=(xs[ci], ys[ci]),
                    xytext=(xs[ci], min(ys[ci] + 0.18, 0.98)),
                    ha="center", arrowprops=dict(arrowstyle="->"))
    ax.text(0.5, -0.16, CURVE_LABEL, transform=ax.transAxes,
            ha="center", fontsize=7, wrap=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def emit_results(curve: list[CurvePoint],
                 spotcheck: list[SpotCheckRow],
                 meta: dict,
                 out_json: str, out_png: str) -> tuple[str, str]:
    """Write the JSON record and the honest PNG. Returns (json_path, png_path)."""
    payload = {
        "meta": meta,
        "labels": {"curve": CURVE_LABEL, "spotcheck": SPOTCHECK_LABEL},
        "curve": [p.model_dump() for p in curve],
        "spotcheck": [r.model_dump() for r in spotcheck],
    }
    Path(out_json).write_text(json.dumps(payload, indent=2))
    _render_png(curve, meta, out_png)
    return out_json, out_png
