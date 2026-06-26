"""
src/ontology.py — Per-clip ontology proposal pass.

Pure module: pydantic + stdlib + src.resolve only.
No torch, no I/O, no LLM imports at module level.
"""

from pydantic import BaseModel

from src.resolve import canonical_relation


# ---------------------------------------------------------------------------
# Ontology model + base vocabulary
# ---------------------------------------------------------------------------

class Ontology(BaseModel):
    entity_types: list[str]
    relations: list[str]


BASE_ONTOLOGY = Ontology(
    entity_types=["Person", "Organization", "Instrument", "Metric", "Decision", "Task", "Topic", "Date"],
    relations=["MENTIONS", "HAS_ATTRIBUTE", "HAS_VALUE", "DECIDED", "COMMITTED_TO", "COMPARES_WITH", "RELATES_TO"],
)

ONTOLOGY_SCHEMA = {           # strict json_schema passed to chat_json
    "type": "object",
    "properties": {
        "entity_types": {"type": "array", "items": {"type": "string"}},
        "relations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["entity_types", "relations"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Ontology proposal
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a knowledge-graph schema assistant. "
    "Given a conversation transcript, propose a SMALL canonical vocabulary "
    "for entity types and relation types that covers the key concepts. "
    "Return a JSON object with 'entity_types' (list of PascalCase strings) "
    "and 'relations' (list of short descriptive phrases or UPPER_SNAKE labels). "
    "Aim for 5-10 entity types and 5-10 relation types. Be specific to the domain."
)


def propose_ontology(transcript_text: str, llm) -> Ontology:
    """
    Propose a per-clip ontology from the transcript via the LLM.

    Falls back to BASE_ONTOLOGY if:
    - the LLM call raises any exception, OR
    - entity_types or relations is empty in the response.

    Relations are deduped and canonicalized via canonical_relation().
    """
    try:
        raw = llm.chat_json(
            _SYSTEM_PROMPT,
            f"Transcript:\n{transcript_text}",
            ONTOLOGY_SCHEMA,
        )
        ont = Ontology(**raw)

        if not ont.entity_types or not ont.relations:
            return BASE_ONTOLOGY

        relations = sorted({canonical_relation(r) for r in ont.relations})
        return Ontology(entity_types=ont.entity_types, relations=relations)

    except Exception:
        return BASE_ONTOLOGY
