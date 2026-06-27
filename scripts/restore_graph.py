"""Deterministically restore the authoritative demo graph from the committed snapshot.
Wipes the graph and recreates the EXACT captured state — the recovery path if Neo4j is
cleared/corrupted before the demo.

Run: source .venv/bin/activate && python -m scripts.restore_graph
"""
import json
import os
from pathlib import Path

from src.config import load_config
from src.graph import connect, restore_graph


def main() -> None:
    cfg = load_config()
    db = os.environ.get("NEO4J_DATABASE", "neo4j")
    snap = json.loads(
        (Path(cfg.paths.ground_truth) / f"{cfg.demo.clip}_graph_snapshot.json").read_text())
    drv = connect()
    try:
        restore_graph(drv, db, snap, wipe=True)
    finally:
        drv.close()
    print(f"restored {len(snap['nodes'])} nodes, {len(snap['rels'])} rels")


if __name__ == "__main__":
    main()
