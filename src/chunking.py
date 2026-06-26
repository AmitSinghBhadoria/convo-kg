"""Turn-boundary transcript chunker with tiktoken token-counting ruler.

Imports only: tiktoken, pydantic, src.contracts — torch-free.
"""
from __future__ import annotations

from typing import Callable

import tiktoken
from pydantic import BaseModel

from src.contracts import Utterance

# ---------------------------------------------------------------------------
# Token counter (ruler only — not used for generation, just sizing)
# ---------------------------------------------------------------------------
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the number of cl100k_base tokens in *text*."""
    return len(_ENCODING.encode(text))


# ---------------------------------------------------------------------------
# Chunk model
# ---------------------------------------------------------------------------
class Chunk(BaseModel):
    index: int
    utterances: list[Utterance]          # turns to extract from
    indices: list[int]                   # their global utterance indices (stable statement ids)
    context: Utterance | None = None      # previous chunk's last turn, read-only overlap
    context_index: int | None = None


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
def chunk_transcript(
    utterances: list[Utterance],
    target_tokens: int,
    count: Callable[[str], int] = count_tokens,
) -> list[Chunk]:
    """Split *utterances* into `Chunk` objects respecting turn boundaries.

    Rules (hard constraints):
    - Never split mid-turn; a turn is the atomic unit.
    - When adding the next turn would push the running total above
      *target_tokens* AND the current chunk is non-empty, cut BEFORE that
      turn (soft target).
    - A single turn larger than *target_tokens* lands alone as an oversized
      chunk (no way to split without violating the mid-turn rule).
    - Each chunk after the first carries the previous chunk's last turn as
      ``context`` (read-only overlap; its global index is ``context_index``).
    """
    if not utterances:
        return []

    chunks: list[Chunk] = []
    current_utts: list[Utterance] = []
    current_indices: list[int] = []
    current_tokens: int = 0

    def _flush(context_utt: Utterance | None, context_idx: int | None) -> None:
        chunks.append(
            Chunk(
                index=len(chunks),
                utterances=list(current_utts),
                indices=list(current_indices),
                context=context_utt,
                context_index=context_idx,
            )
        )
        current_utts.clear()
        current_indices.clear()

    last_context_utt: Utterance | None = None
    last_context_idx: int | None = None
    # After each flush the very first turn of the new chunk is always admitted
    # without an overflow check (this is how oversized single turns land alone,
    # and how the greedy packing avoids degenerate one-turn-per-chunk splits).
    just_flushed: bool = True

    for i, utt in enumerate(utterances):
        turn_tokens = count(utt.text)

        would_exceed = (current_tokens + turn_tokens) > target_tokens

        if (not just_flushed) and would_exceed and current_utts:
            # Record context for the NEXT chunk before flushing
            next_context_utt = current_utts[-1]
            next_context_idx = current_indices[-1]
            _flush(last_context_utt, last_context_idx)
            current_tokens = 0
            last_context_utt = next_context_utt
            last_context_idx = next_context_idx
            just_flushed = True
        else:
            just_flushed = False

        current_utts.append(utt)
        current_indices.append(i)
        current_tokens += turn_tokens

    # Flush whatever remains
    if current_utts:
        _flush(last_context_utt, last_context_idx)

    return chunks
