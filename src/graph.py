"""
src/graph.py — Idempotent Neo4j upsert for FactSet + Transcript.

Pure stdlib + dotenv + neo4j driver.  No torch/numpy.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j import Driver

from src.contracts import FactSet, Transcript
from src.resolve import safe_rel_type, slugify, statement_id as make_statement_id

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKBONE_LABELS: frozenset[str] = frozenset(
    {"Speaker", "Statement", "Entity", "Claim", "Attribute"}
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flatten_props(d: dict) -> dict:
    """
    Return a copy of *d* safe for Neo4j property storage.

    Primitives (str, int, float, bool, None) are kept as-is.
    Anything else (list, dict, …) is JSON-serialised to a string.
    """
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
    return out


def _entity_merge_cypher(label: str) -> str:
    """
    Return a Cypher MERGE fragment for the given entity label.

    Raises ValueError if *label* is not in BACKBONE_LABELS (injection guard).
    The label is backtick-interpolated only after validation.
    """
    if label not in BACKBONE_LABELS:
        raise ValueError(
            f"Label {label!r} is not in the backbone allowlist {set(BACKBONE_LABELS)!r}. "
            "Use one of: Speaker, Statement, Entity, Claim, Attribute."
        )
    return (
        f"MERGE (n:`{label}` {{id:$id}}) "
        f"SET n.name=$name, n.type=$type, n += $attrs"
    )


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect() -> Driver:
    """
    Open a Neo4j driver from .env credentials.

    Raises RuntimeError with a clear hint if any credential is missing or if
    the instance is unreachable.
    """
    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not uri or not username or not password:
        raise RuntimeError(
            "Missing NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD in .env — "
            "is the 'atyx' instance running?"
        )

    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        return driver
    except Exception as exc:
        raise RuntimeError(
            f"Cannot connect to Neo4j at {uri!r} — "
            "is the 'atyx' instance running?"
        ) from exc


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert(
    factset: FactSet,
    transcript: Transcript,
    driver: Driver,
    database: str = "neo4j",
) -> dict:
    """
    Idempotently upsert *transcript* statements + *factset* entities/facts into Neo4j.

    Returns dict with keys 'statements', 'entities', 'facts' (counts written).

    All Cypher values are parameterized; labels and relationship types are
    validated before string interpolation.
    """
    clip = transcript.clip
    stmt_count = 0
    entity_count = 0
    fact_count = 0

    with driver.session(database=database) as session:

        # ------------------------------------------------------------------
        # 1. Statements + Speakers (one per transcript utterance)
        # ------------------------------------------------------------------
        for i, utt in enumerate(transcript.utterances):
            sid = make_statement_id(clip, i)
            spk = utt.speaker
            pid = f"speaker:{slugify(spk)}"

            def _write_statement(tx, sid=sid, spk=spk, pid=pid, utt=utt, clip=clip):
                tx.run(
                    "MERGE (s:Statement {id:$id}) "
                    "SET s.text=$text, s.speaker=$spk, s.clip=$clip, s.start=$start, s.end=$end",
                    id=sid, text=utt.text, spk=spk, clip=clip,
                    start=utt.start, end=utt.end,
                )
                tx.run(
                    "MERGE (p:Speaker {id:$pid}) SET p.name=$spk",
                    pid=pid, spk=spk,
                )
                tx.run(
                    "MATCH (p:Speaker {id:$pid}), (s:Statement {id:$sid}) "
                    "MERGE (p)-[:SAID]->(s)",
                    pid=pid, sid=sid,
                )

            session.execute_write(_write_statement)
            stmt_count += 1

        # ------------------------------------------------------------------
        # 2. Entities
        # ------------------------------------------------------------------
        for ent in factset.entities:
            cypher = _entity_merge_cypher(ent.label)  # raises ValueError on bad label
            attrs = flatten_props(ent.attrs)

            def _write_entity(tx, cypher=cypher, ent=ent, attrs=attrs):
                tx.run(cypher, id=ent.id, name=ent.name, type=ent.type, attrs=attrs)

            session.execute_write(_write_entity)
            entity_count += 1

        # ------------------------------------------------------------------
        # 3. Fact edges
        # ------------------------------------------------------------------
        for fact in factset.facts:
            rel = safe_rel_type(fact.relation)  # raises ValueError on bad rel type
            cypher = (
                f"MATCH (a {{id:$sid}}), (b {{id:$oid}}) "
                f"MERGE (a)-[r:`{rel}`]->(b) "
                f"SET r.confidence=$conf, r.speaker=$spk, "
                f"r.source_statement_id=$ssid, r.statement=$stmt"
            )

            def _write_fact(tx, cypher=cypher, fact=fact):
                tx.run(
                    cypher,
                    sid=fact.subject_id,
                    oid=fact.object_id,
                    conf=fact.confidence,
                    spk=fact.speaker,
                    ssid=fact.statement_id,
                    stmt=fact.statement,
                )

            session.execute_write(_write_fact)
            fact_count += 1

    return {"statements": stmt_count, "entities": entity_count, "facts": fact_count}


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

def run(clip: str) -> dict:
    """
    Load <clip>.facts.json + <clip>.transcript.json, upsert into Neo4j.

    Database taken from NEO4J_DATABASE env var (default: 'neo4j').
    Prints counts and returns them.
    """
    from src.config import load_config
    load_dotenv()
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    # Read artifacts from the same work dir extract.py writes them to (cfg.paths.work).
    work = Path(load_config().paths.work)
    facts_path = work / f"{clip}.facts.json"
    transcript_path = work / f"{clip}.transcript.json"

    if not facts_path.exists():
        raise FileNotFoundError(f"Facts file not found: {facts_path}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    factset = FactSet.model_validate_json(facts_path.read_text())
    transcript = Transcript.model_validate_json(transcript_path.read_text())

    driver = connect()
    try:
        counts = upsert(factset, transcript, driver, database=database)
    finally:
        driver.close()

    print(
        f"Upserted  statements={counts['statements']}  "
        f"entities={counts['entities']}  "
        f"facts={counts['facts']}"
    )
    return counts


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.graph <clip>", file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1])
