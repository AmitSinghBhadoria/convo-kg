"""
src/resolve.py — Identity + precision-biased entity/relation resolution.

Pure module: stdlib (re, math) + src.contracts only.
No I/O, no LLM calls, no torch/numpy.
"""

import re
import math
from typing import Callable

from src.contracts import Entity

# ---------------------------------------------------------------------------
# Low-level text utilities
# ---------------------------------------------------------------------------

def normalize_name(s: str) -> str:
    """Lowercase, strip leading/trailing whitespace, collapse internal whitespace."""
    return re.sub(r'\s+', ' ', s.strip()).lower()


def slugify(s: str) -> str:
    """Lowercase; strip non-alphanumeric, non-space chars; replace spaces with hyphens."""
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = s.strip()
    s = re.sub(r'\s+', '-', s)
    return s


def entity_id(label: str, name: str) -> str:
    """Stable dedupe key: '<label_lower>:<slugify(name)>'."""
    return f"{label.lower()}:{slugify(name)}"


def statement_id(clip: str, idx: int) -> str:
    """Stable statement key: 'stmt:<clip>:<idx>'."""
    return f"stmt:{clip}:{idx}"


# ---------------------------------------------------------------------------
# Relation canonicalization
# ---------------------------------------------------------------------------

# Abbreviation expansions (whole-word only via \b anchors).
_ABBREV: dict[str, str] = {
    'min': 'minimum',
    'max': 'maximum',
    'avg': 'average',
    'amt': 'amount',
    'num': 'number',
    'qty': 'quantity',
    'pct': 'percent',
    'yr':  'year',
    'yrs': 'years',
    'mo':  'month',
    'mos': 'months',
}

# Pre-compile abbreviation regex once (longest-first ordering avoids partial shadow).
_ABBREV_RE = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in _ABBREV) + r')\b'
)

# Minimal, conservative stop-word set — ONLY these five.
# Do NOT add prepositions (to, on, with) or generic-but-meaningful nouns (value, number).
_STOP_WORDS: frozenset[str] = frozenset({'a', 'an', 'the', 'of', 'amount'})

# Pattern for an already-canonical UPPER_SNAKE relation type.
_UPPER_SNAKE_RE = re.compile(r'^[A-Z][A-Z0-9_]*\Z')  # \Z, not $: a trailing newline must NOT pass


def canonical_relation(relation: str, vocab=None) -> str:
    """
    Canonicalize a relation string to a valid Neo4j rel type.

    Contract:
      • Idempotency guard FIRST: if `relation` already matches ^[A-Z][A-Z0-9_]*$
        it is returned unchanged — every BASE_ONTOLOGY entry round-trips to itself.
      • Otherwise:
          1. lowercase
          2. expand abbreviations (whole-word: min→minimum, max→maximum, …)
          3. strip punctuation (replace [^a-z0-9\\s] with space)
          4. split → drop stop words {a, an, the, of, amount} ONLY
             (prepositions to/on/with and nouns value/number/… are intentionally kept)
          5. join with '_', UPPER, strip to [A-Z0-9_]
      • If `vocab` is given and the result is already in it, return it.
    """
    # (0) Idempotency guard — must be checked BEFORE any transformation.
    if _UPPER_SNAKE_RE.match(relation):
        return relation

    # (1) Lowercase.
    s = relation.lower()

    # (2) Expand abbreviations (whole-word substitution).
    s = _ABBREV_RE.sub(lambda m: _ABBREV[m.group(1)], s)

    # (3) Strip punctuation — replace non-alpha/non-digit/non-space with a space.
    s = re.sub(r'[^a-z0-9\s]', ' ', s)

    # (4) Tokenize and drop stop words.
    tokens = [t for t in s.split() if t and t not in _STOP_WORDS]

    # (5) Join with '_', UPPER, safety-strip residual non-[A-Z0-9_] chars.
    result = '_'.join(tokens).upper()
    result = re.sub(r'[^A-Z0-9_]', '', result)

    # (6) If vocab provided and result is in it, return canonically.
    if vocab is not None and result in vocab:
        return result

    return result


def safe_rel_type(rel: str) -> str:
    """Return `rel` if it is a valid Neo4j relationship type, else raise ValueError."""
    if _UPPER_SNAKE_RE.match(rel):
        return rel
    raise ValueError(
        f"Relation type {rel!r} is not a valid Neo4j rel type "
        f"(must match ^[A-Z][A-Z0-9_]*$)"
    )


# ---------------------------------------------------------------------------
# Cosine similarity — pure Python, no numpy/torch dependency.
# ---------------------------------------------------------------------------

def _cosine(u: list[float], v: list[float]) -> float:
    """Cosine similarity between two vectors (returns 0.0 if either is zero-length)."""
    dot = sum(a * b for a, b in zip(u, v))
    norm_u = math.sqrt(sum(a * a for a in u))
    norm_v = math.sqrt(sum(b * b for b in v))
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return dot / (norm_u * norm_v)


# ---------------------------------------------------------------------------
# EntityResolver — precision-biased deduplication
# ---------------------------------------------------------------------------

class EntityResolver:
    """
    Precision-biased entity resolver.

    Merges entities only when confident:
      1. Exact normalized-name match (same label AND type) → reuse representative.
      2. Embedding fallback among existing reps with the SAME label AND type:
         merge only if cosine similarity ≥ `threshold` (default 0.85).
      3. Otherwise: new representative with id = entity_id(label, name).

    When uncertain, do NOT merge.

    Parameters
    ----------
    embed_fn : callable(list[str]) -> list[list[float]]
        Injectable embedding function; receives a list of name strings, returns
        a parallel list of float vectors. Tests pass fakes here.
    threshold : float
        Cosine similarity threshold for embedding-based merging (default 0.85).
    """

    def __init__(
        self,
        embed_fn: Callable[[list[str]], list[list[float]]],
        threshold: float = 0.85,
    ) -> None:
        self.embed_fn = embed_fn
        self.threshold = threshold

    def resolve(
        self, entities: list[Entity]
    ) -> tuple[list[Entity], dict[str, str]]:
        """
        Resolve a list of entities into canonical representatives.

        Returns
        -------
        reps : list[Entity]
            Deduplicated representative entities with stable ids.
        idmap : dict[str, str]
            Maps every input entity id to its representative's id.
        """
        reps: list[Entity] = []
        idmap: dict[str, str] = {}

        # Fast lookup: (label, type, normalized_name) → representative Entity.
        name_index: dict[tuple[str, str, str], Entity] = {}

        for ent in entities:
            norm = normalize_name(ent.name)
            key = (ent.label, ent.type, norm)

            # (1) Exact normalized-name match — highest confidence, no embedding needed.
            if key in name_index:
                rep = name_index[key]
                idmap[ent.id] = rep.id
                continue

            # (2) Embedding fallback — restrict to same label AND type (precision guard).
            candidates = [r for r in reps if r.label == ent.label and r.type == ent.type]
            if candidates:
                cand_names = [r.name for r in candidates]
                all_names = cand_names + [ent.name]
                vecs = self.embed_fn(all_names)
                cand_vecs = vecs[:-1]
                ent_vec = vecs[-1]

                best_score = -1.0
                best_rep: Entity | None = None
                for rep, cvec in zip(candidates, cand_vecs):
                    score = _cosine(ent_vec, cvec)
                    if score >= self.threshold and score > best_score:
                        best_score = score
                        best_rep = rep

                if best_rep is not None:
                    idmap[ent.id] = best_rep.id
                    # Cache normalized name → same rep so future exact matches are O(1).
                    name_index[key] = best_rep
                    continue

            # (3) New representative — assign a stable id.
            new_id = entity_id(ent.label, ent.name)
            rep = ent.model_copy(update={"id": new_id})
            reps.append(rep)
            idmap[ent.id] = new_id
            name_index[key] = rep

        return reps, idmap
