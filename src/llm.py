import json
from openai import OpenAI
from src.config import LLMCfg

class LLM:
    def __init__(self, cfg: LLMCfg):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key="lm-studio")

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        rf = {"type": "json_schema",
              "json_schema": {"name": "out", "strict": True, "schema": schema}}
        for attempt in range(2):
            r = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format=rf, temperature=0)
            try:
                return json.loads(r.choices[0].message.content)
            except json.JSONDecodeError:
                if attempt == 1: raise
                user = user + "\n\nReturn ONLY valid JSON matching the schema."

    def embed(self, texts: list[str]) -> list[list[float]]:
        r = self.client.embeddings.create(model=self.cfg.embed_model, input=texts)
        return [d.embedding for d in r.data]
