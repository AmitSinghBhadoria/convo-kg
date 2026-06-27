import json
from openai import OpenAI
from src.config import LLMCfg

class LLM:
    def __init__(self, cfg: LLMCfg):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key="lm-studio")

    def chat_json(self, system: str, user: str, schema: dict, max_tokens: int = 2048) -> dict:
        rf = {"type": "json_schema",
              "json_schema": {"name": "out", "strict": True, "schema": schema}}
        for attempt in range(2):
            r = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                # max_tokens is a safety bound: it caps the blast radius of any
                # runaway/repetition generation (a single call can no longer burn
                # ~9k tokens). The real fix for the extraction loop is maxItems on
                # the schema arrays; this is defense in depth. A clean extraction
                # JSON for one chunk is well under this, so it does not truncate
                # legitimate output — verified on the dense PMS chunk.
                response_format=rf, temperature=0, max_tokens=max_tokens)
            msg = r.choices[0].message
            # Thinking models (e.g. Qwen3.5) may route the JSON to reasoning_content
            # and leave content empty; fall back to it so we're robust to the toggle state.
            raw = msg.content or getattr(msg, "reasoning_content", None) or ""
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                if attempt == 1: raise
                user = user + "\n\nReturn ONLY valid JSON matching the schema."

    def embed(self, texts: list[str]) -> list[list[float]]:
        r = self.client.embeddings.create(model=self.cfg.embed_model, input=texts)
        return [d.embedding for d in r.data]
