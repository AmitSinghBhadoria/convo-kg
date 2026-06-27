"""Phase 4 controlled-SNR evaluation harness.

Pure/IO split: snr_curve + downstream_spotcheck + retrieve_answer are pure /
LLM-injected and unit-tested with a fake; emit_results does IO (JSON + PNG);
evaluate() orchestrates over the sweep transcripts already on disk.

Hero curve = transcript similarity vs SNR, oracle-free (each noisy transcript
scored against the clean slice through the identical pipeline). The downstream
spot-check is transcript-grounded retrieval (NOT the full extract->graph->Q&A
product path) — see docs/superpowers/specs/2026-06-27-atyx-convo-kg-phase4-eval-design.md.
"""
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
