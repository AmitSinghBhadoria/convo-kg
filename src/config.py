from pathlib import Path
import os, yaml
from pydantic import BaseModel

class LLMCfg(BaseModel):
    base_url: str
    model: str
    embed_model: str

class Paths(BaseModel):
    raw: str; work: str; noisy: str; ground_truth: str

class ExtractCfg(BaseModel):
    chunk_tokens: int; overlap_tokens: int; confidence_threshold: float

class EvalCfg(BaseModel):
    snr_levels: list[int]

class Limits(BaseModel):
    max_minutes: int

class Config(BaseModel):
    llm: LLMCfg; paths: Paths; extract: ExtractCfg; eval: EvalCfg; limits: Limits

def load_config(path: str = "config.yaml") -> Config:
    data = yaml.safe_load(Path(path).read_text())
    if v := os.getenv("LLM_BASE_URL"): data["llm"]["base_url"] = v
    if v := os.getenv("LLM_MODEL"): data["llm"]["model"] = v
    return Config.model_validate(data)
