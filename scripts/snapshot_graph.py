"""Capture the EXACT live Neo4j graph to a committed JSON file. The authoritative demo
graph is the crown jewel; extraction is nondeterministic, so this snapshot — not a
re-run of extract — is the source of truth for restore.

Run: source .venv/bin/activate && python -m scripts.snapshot_graph
"""
import json
import os
from pathlib import Path

from src.config import load_config
from src.graph import connect, export_graph


def main() -> None:
    cfg = load_config()
    db = os.environ.get("NEO4J_DATABASE", "neo4j")
    drv = connect()
    try:
        snap = export_graph(drv, db)
    finally:
        drv.close()
    out = Path(cfg.paths.ground_truth) / f"{cfg.demo.clip}_graph_snapshot.json"
    out.write_text(json.dumps(snap, indent=2))
    print(f"snapshot: {len(snap['nodes'])} nodes, {len(snap['rels'])} rels -> {out}")


if __name__ == "__main__":
    main()
