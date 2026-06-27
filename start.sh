#!/usr/bin/env bash
# Atyx Convo-KG — start the demo.
#
# Preflights Neo4j + LM Studio, ensures the verified demo graph is loaded (from
# the committed snapshot), then serves the FastAPI app and opens the UI.
#
#   ./start.sh             start the demo (restores the graph only if empty)
#   ./start.sh --restore   force-restore the demo graph from the snapshot first
#
# Needs: ./setup.sh already run; Neo4j Desktop + LM Studio running.
set -euo pipefail
cd "$(dirname "$0")"

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32mok\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mxx %s\033[0m\n' "$*"; exit 1; }

FORCE_RESTORE=0
[ "${1:-}" = "--restore" ] && FORCE_RESTORE=1

# ---------------------------------------------------------------------------
[ -d .venv ] || die "no .venv — run ./setup.sh first"
# Export .env so NEO4J_DATABASE etc. are present at import time.
[ -f .env ] && { set -a; . ./.env; set +a; }
# shellcheck disable=SC1091
source .venv/bin/activate
ok "main venv active ($(python --version 2>&1 | awk '{print $2}'))"

# ---------------------------------------------------------------------------
say "Preflight — Neo4j"
python - <<'PY' || die "Neo4j not reachable — Start your Neo4j Desktop instance and check .env (NEO4J_URI/USERNAME/PASSWORD)."
from src.graph import connect
connect().close()
print("ok  Neo4j reachable")
PY

say "Preflight — LM Studio (local LLM)"
LLM_URL="$(python -c 'from src.config import load_config; print(load_config().llm.base_url)')"
if curl -sf "${LLM_URL}/models" >/dev/null 2>&1; then
  ok "LM Studio reachable at ${LLM_URL}"
else
  warn "LM Studio NOT reachable at ${LLM_URL}"
  warn "Ask-Atyx and live Run need it — load qwen/qwen3.5-9b + nomic embed (Thinking OFF)."
  warn "Graph + Experiment tabs still work without it. Continuing anyway."
fi

# ---------------------------------------------------------------------------
say "Demo graph"
NODES="$(python -c 'import os; from src.graph import connect; d=connect(); db=os.environ.get("NEO4J_DATABASE","neo4j"); s=d.session(database=db); n=s.run("MATCH (n) RETURN count(n) AS n").single()["n"]; print(n); d.close()')"
if [ "$FORCE_RESTORE" = "1" ] || [ "$NODES" = "0" ]; then
  warn "restoring the verified demo graph from data/ground_truth/*_graph_snapshot.json"
  python -m scripts.restore_graph
else
  ok "graph present ($NODES nodes) — use ./start.sh --restore to reset it to the verified snapshot"
fi

# ---------------------------------------------------------------------------
say "Serving — http://localhost:8000  (Ctrl-C to stop)"
( sleep 2; (command -v open >/dev/null && open http://localhost:8000) || true ) &
exec python -m src.api
