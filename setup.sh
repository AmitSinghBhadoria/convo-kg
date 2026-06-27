#!/usr/bin/env bash
# Atyx Convo-KG — one-shot repo setup.
#
# Creates the Python environments and checks prerequisites so a fresh clone can
# run the demo. The audio pipeline uses isolated venvs (irreconcilable torch
# pins); the demo itself needs only the main .venv + Neo4j + LM Studio.
#
#   ./setup.sh              full setup (main + ASR + denoise venvs)
#   SKIP_AUDIO=1 ./setup.sh main venv only — enough for the demo (Ask/Graph/
#                           Experiment + Run replay); skips the heavy torch venvs
#
# Prerequisites you must install/run yourself (external apps, can't be scripted):
#   - uv            (https://docs.astral.sh/uv/)   — Python env manager
#   - Neo4j Desktop — a running local instance, creds in .env
#   - LM Studio     — qwen/qwen3.5-9b + nomic-embed loaded, Reasoning/Thinking OFF
set -euo pipefail
cd "$(dirname "$0")"

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32mok\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------------------
say "Checking prerequisites"
if ! command -v uv >/dev/null 2>&1; then
  warn "uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
ok "uv $(uv --version | awk '{print $2}')"

# ---------------------------------------------------------------------------
say "Main environment (.venv, Python 3.12, torch-free) — required for the demo"
uv venv --python 3.12 .venv
uv sync
ok "main deps installed (FastAPI, neo4j, openai, pydantic, ...)"

# ---------------------------------------------------------------------------
if [ "${SKIP_AUDIO:-0}" = "1" ]; then
  warn "SKIP_AUDIO=1 — skipping ASR/denoise venvs (only needed to regenerate audio/eval artifacts)"
else
  say "ASR environment (.venv-asr, Python 3.12) — whisper translate + pyannote (large, torch)"
  uv venv .venv-asr --python 3.12
  uv pip install --python .venv-asr -r requirements-asr.txt
  ok ".venv-asr ready"

  say "Denoise environment (.venv-denoise, Python 3.11) — DeepFilterNet (large, torch)"
  uv venv .venv-denoise --python 3.11
  uv pip install --python .venv-denoise -r requirements-denoise.txt
  ok ".venv-denoise ready"
fi

# ---------------------------------------------------------------------------
say "Secrets (.env)"
if [ -f .env ]; then
  ok ".env present"
else
  cat > .env <<'ENV'
# Neo4j Desktop — create a local instance, Start it, put its password here
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=CHANGE_ME
NEO4J_DATABASE=neo4j
# Hugging Face token (only needed for the audio pipeline: pyannote diarization)
HF_TOKEN=hf_CHANGE_ME
ENV
  warn "wrote a template .env — edit NEO4J_PASSWORD (and HF_TOKEN for the audio pipeline)"
fi

# ---------------------------------------------------------------------------
say "Setup complete"
cat <<'NEXT'
Before running the demo, make sure these external services are up:
  1. Neo4j Desktop — your local instance is Started, password matches .env
  2. LM Studio     — qwen/qwen3.5-9b + the nomic embedding model loaded,
                     and Reasoning/Thinking turned OFF

Then start the app:
  ./start.sh

start.sh loads the verified demo graph from the committed snapshot, runs the
FastAPI server, and opens the UI at http://localhost:8000
NEXT
