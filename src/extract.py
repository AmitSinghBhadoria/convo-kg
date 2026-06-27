"""
src/extract.py — Extraction pass: chunks → LLM → consolidated FactSet.

Imports: stdlib + src.* only. No torch.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from src.contracts import Entity, Fact, FactSet, Transcript
from src.chunking import Chunk, chunk_transcript
from src.resolve import EntityResolver, canonical_relation, statement_id
from src.ontology import Ontology, propose_ontology


# ---------------------------------------------------------------------------
# EXTRACT_SCHEMA — strict json_schema for one chunk's LLM output
# ---------------------------------------------------------------------------

EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            # maxItems bounds the constrained-decoding grammar so the array is
            # forced to close — prevents the greedy-decode repetition loop that
            # otherwise lets a dense chunk emit the same objects unboundedly.
            # 15 is generous headroom for one small (~700-token) chunk and is
            # kept consistent with llm.max_tokens: at ~100 tokens per item, the
            # closed array fits under the 2048 cap with margin, so the grammar
            # closes the array naturally instead of the token cap truncating it.
            "maxItems": 15,
            "items": {
                "type": "object",
                "properties": {
                    "id":    {"type": "string"},
                    "label": {
                        "type": "string",
                        # Extraction may ONLY emit content nodes. Speaker and
                        # Statement are backbone nodes created from the transcript
                        # by graph.upsert (clean, diarization-derived); letting the
                        # LLM emit them spawns duplicate/garbage Speaker nodes
                        # ("Viraj", "Speaker_02", "SPEAKER_02") that collide with
                        # the real ones and corrupt speaker attribution.
                        "enum": ["Entity", "Claim", "Attribute"],
                    },
                    "type": {"type": "string"},
                    "name": {"type": "string"},
                },
                "required": ["id", "label", "type", "name"],
                "additionalProperties": False,
            },
        },
        "facts": {
            "type": "array",
            # See the entities note: bounds the array so the grammar closes it,
            # kept consistent with llm.max_tokens (~100 tokens/fact -> 15 facts
            # fit under the 2048 cap with margin). With small chunks the array
            # closes naturally well below this; it is a backstop, not a target.
            "maxItems": 15,
            "items": {
                "type": "object",
                "properties": {
                    "subject_id":   {"type": "string"},
                    "relation":     {"type": "string"},
                    "object_id":    {"type": "string"},
                    "statement":    {"type": "string"},
                    "statement_id": {"type": "string"},
                    "speaker":      {"type": "string"},
                    "confidence":   {"type": "number"},
                },
                "required": [
                    "subject_id", "relation", "object_id",
                    "statement", "statement_id", "speaker", "confidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities", "facts"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

def build_prompt(
    chunk: Chunk,
    ontology: Ontology | None,
    clip: str,
) -> tuple[str, str]:
    """Return (system, user) prompt for extracting facts from a single chunk."""
    # --- System prompt ---
    if ontology is not None:
        schema_hint = (
            f"Bias your extraction toward these types:\n"
            f"Entity types: {', '.join(ontology.entity_types)}.\n"
            f"Relation types: {', '.join(ontology.relations)}."
        )
    else:
        schema_hint = "Induce sensible entity and relation types from the conversation."

    system = (
        "You are a knowledge-graph extraction assistant.\n"
        f"{schema_hint}\n\n"
        "Rules:\n"
        "1. Every fact MUST set `statement_id` to the bracketed tag of its source line "
        "(e.g. [stmt:clip:5] → statement_id is 'stmt:clip:5').\n"
        "2. Use entity ids you define (short local identifiers, e.g. 'e1', 'e2').\n"
        "3. Only extract facts that are EXPLICITLY stated in the conversation.\n"
        "4. Assign `confidence` in [0, 1] for each fact.\n"
        "5. NEVER extract from or cite lines marked "
        "(CONTEXT-ONLY — do not extract, do not cite).\n"
        "6. Return valid JSON matching the schema exactly."
    )

    # --- User prompt: optional context line + extractable turns ---
    lines: list[str] = []

    if chunk.context is not None and chunk.context_index is not None:
        ctx_sid = statement_id(clip, chunk.context_index)
        lines.append(
            f"[{ctx_sid}] {chunk.context.speaker}: {chunk.context.text} "
            f"(CONTEXT-ONLY — do not extract, do not cite)"
        )

    for utt, idx in zip(chunk.utterances, chunk.indices):
        sid = statement_id(clip, idx)
        lines.append(f"[{sid}] {utt.speaker}: {utt.text}")

    user = "\n".join(lines)
    return system, user


# ---------------------------------------------------------------------------
# namespace — pure
# ---------------------------------------------------------------------------

def namespace(
    entities: list[Entity],
    facts: list[Fact],
    index: int,
) -> tuple[list[Entity], list[Fact]]:
    """Prefix every entity id and every fact subject_id/object_id with f'c{index}:'."""
    prefix = f"c{index}:"
    new_entities = [
        ent.model_copy(update={"id": f"{prefix}{ent.id}"})
        for ent in entities
    ]
    new_facts = [
        fact.model_copy(update={
            "subject_id": f"{prefix}{fact.subject_id}",
            "object_id":  f"{prefix}{fact.object_id}",
        })
        for fact in facts
    ]
    return new_entities, new_facts


# ---------------------------------------------------------------------------
# _extract_chunk (internal)
# ---------------------------------------------------------------------------

def _extract_chunk(
    chunk: Chunk,
    ontology: Ontology | None,
    clip: str,
    llm,
) -> tuple[list[Entity], list[Fact]]:
    """Call LLM for one chunk; return namespaced (entities, facts).

    On ANY error: log to stderr and return ([], []) — partial graph > crash.
    """
    system, user = build_prompt(chunk, ontology, clip)
    try:
        raw = llm.chat_json(system, user, EXTRACT_SCHEMA)
        entities = [Entity(**e) for e in raw.get("entities", [])]
        facts    = [Fact(**f)   for f in raw.get("facts", [])]
    except Exception as exc:
        print(f"[extract] chunk {chunk.index} error: {exc}", file=sys.stderr)
        return [], []

    return namespace(entities, facts, chunk.index)


# ---------------------------------------------------------------------------
# consolidate
# ---------------------------------------------------------------------------

def consolidate(
    raw_entities: list[Entity],
    raw_facts: list[Fact],
    vocab: set[str] | None,
    threshold: float,
    resolver: EntityResolver,
    clip: str,
) -> FactSet:
    """Merge entities, filter + dedup facts, return a grounded FactSet."""
    # Defensive backstop to the schema enum: extraction must never create
    # backbone nodes. Drop any Speaker/Statement-labeled entity so a stray one
    # cannot pollute the graph's diarization-derived Speaker/Statement nodes.
    # Facts referencing a dropped entity fall away via the dangling-ref guard.
    raw_entities = [e for e in raw_entities if e.label not in ("Speaker", "Statement")]
    reps, id_map = resolver.resolve(raw_entities)
    rep_ids = {e.id for e in reps}

    seen: set[tuple[str, str, str]] = set()
    facts: list[Fact] = []

    for fact in raw_facts:
        # Drop ungrounded facts (grounding is mandatory; "" is a valid string for the
        # schema, so enforce the invariant in code, not only via schema/prompt).
        if not fact.statement_id:
            continue

        # Drop low-confidence (strict < so == threshold is KEPT)
        if fact.confidence < threshold:
            continue

        # Remap through id_map; drop if either end is dangling/ungrounded
        subj = id_map.get(fact.subject_id)
        obj  = id_map.get(fact.object_id)
        if subj is None or obj is None:
            continue
        if subj not in rep_ids or obj not in rep_ids:
            continue

        # Canonicalize relation
        rel = canonical_relation(fact.relation, vocab)

        # Dedup on (subject_id, relation, object_id)
        key = (subj, rel, obj)
        if key in seen:
            continue
        seen.add(key)

        facts.append(fact.model_copy(update={
            "subject_id": subj,
            "object_id":  obj,
            "relation":   rel,
        }))

    return FactSet(clip=clip, entities=reps, facts=facts)


# ---------------------------------------------------------------------------
# extract — top-level entry point
# ---------------------------------------------------------------------------

def extract(clip: str, cfg=None, llm=None) -> FactSet:
    """Run the full extraction pipeline for one clip.

    transcript → propose_ontology → chunk → _extract_chunk × N
    → consolidate → write data/work/<clip>.facts.json
    """
    from src.config import load_config
    from src.llm import LLM as LLMClass

    if cfg is None:
        cfg = load_config()
    if llm is None:
        llm = LLMClass(cfg.llm)

    # Load transcript
    transcript_path = Path(cfg.paths.work) / f"{clip}.transcript.json"
    transcript = Transcript.model_validate_json(transcript_path.read_text())

    # Full text for ontology proposal
    full_text = "\n".join(
        f"{u.speaker}: {u.text}" for u in transcript.utterances
    )

    # Propose ontology (falls back to BASE_ONTOLOGY on any error)
    ontology = propose_ontology(full_text, llm)

    # Chunk
    chunks = chunk_transcript(transcript.utterances, cfg.extract.chunk_tokens)

    # Extract per chunk
    all_entities: list[Entity] = []
    all_facts: list[Fact] = []
    for chunk in chunks:
        ents, facts = _extract_chunk(chunk, ontology, clip, llm)
        all_entities.extend(ents)
        all_facts.extend(facts)

    # Consolidate
    resolver = EntityResolver(embed_fn=llm.embed)
    factset = consolidate(
        all_entities,
        all_facts,
        vocab=set(ontology.relations),
        threshold=cfg.extract.confidence_threshold,
        resolver=resolver,
        clip=clip,
    )

    # Write output
    out_path = Path(cfg.paths.work) / f"{clip}.facts.json"
    out_path.write_text(factset.model_dump_json(indent=2))

    return factset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.extract <clip>", file=sys.stderr)
        sys.exit(1)
    clip_name = sys.argv[1]
    fs = extract(clip_name)
    print(
        f"Extracted {len(fs.entities)} entities, {len(fs.facts)} facts "
        f"→ data/work/{clip_name}.facts.json"
    )
