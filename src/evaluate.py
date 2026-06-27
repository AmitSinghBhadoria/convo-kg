"""Phase 4 controlled-SNR evaluation harness.

Pure/IO split: snr_curve + downstream_spotcheck + retrieve_answer are pure /
LLM-injected and unit-tested with a fake; emit_results does IO (JSON + PNG);
evaluate() orchestrates over the sweep transcripts already on disk.

Hero curve = transcript similarity vs SNR, oracle-free (each noisy transcript
scored against the clean slice through the identical pipeline). The downstream
spot-check is transcript-grounded retrieval (NOT the full extract->graph->Q&A
product path) — see docs/superpowers/specs/2026-06-27-atyx-convo-kg-phase4-eval-design.md.
"""
from src.contracts import CurvePoint, Transcript
from src.evaltools import similarity, transcript_text


def snr_curve(baseline: Transcript,
              noisy: dict[str, Transcript]) -> list[CurvePoint]:
    """Score each noisy transcript against the baseline transcript.

    Oracle-free: the baseline is the clean slice through the identical pipeline.
    Order of points follows the insertion order of `noisy`.
    """
    ref = transcript_text(baseline)
    return [CurvePoint(snr=snr, similarity=similarity(transcript_text(t), ref))
            for snr, t in noisy.items()]
