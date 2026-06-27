import pytest
from src.config import load_config


def test_demo_clip_config_defaults_to_pms():
    assert load_config().demo.clip == "pms"


@pytest.mark.integration
def test_snapshot_restore_roundtrips_exact_counts():
    import os
    from src.graph import connect, export_graph, restore_graph
    drv = connect(); db = os.environ.get("NEO4J_DATABASE", "neo4j")
    try:
        before = export_graph(drv, db)
        assert before["nodes"] and before["rels"]            # graph is non-empty
        restore_graph(drv, db, before, wipe=True)            # wipe + restore the same snapshot
        after = export_graph(drv, db)
        assert len(after["nodes"]) == len(before["nodes"])   # exact node count preserved
        assert len(after["rels"]) == len(before["rels"])     # exact rel count preserved
        assert {n["id"] for n in after["nodes"]} == {n["id"] for n in before["nodes"]}
    finally:
        drv.close()
