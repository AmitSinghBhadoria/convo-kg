from pathlib import Path
import os, yaml
from pydantic import BaseModel, Field

class ClipCfg(BaseModel):
    id: str
    label: str
    mode: str            # "graph" | "facts" | "live"
    domain: str = ""
    speakers: int = 0

class LLMCfg(BaseModel):
    base_url: str
    model: str
    embed_model: str

class Paths(BaseModel):
    raw: str; work: str; noisy: str; ground_truth: str; uploads: str

class ExtractCfg(BaseModel):
    chunk_tokens: int; overlap_tokens: int; confidence_threshold: float

class EvalCfg(BaseModel):
    snr_levels: list[int]
    source_clip: str = "pms"
    slice_start_s: int = 0
    slice_end_s: int = 160
    noise_path: str = "noices/cafe_16k.wav"
    clip_prefix: str = "pmsslice"
    degraded_snr: int = 5
    spotcheck_questions: list[str] = []

class Limits(BaseModel):
    max_minutes: int

class DemoCfg(BaseModel):
    clip: str = "pms"
    clips: list[ClipCfg] = Field(default_factory=list)

class Config(BaseModel):
    llm: LLMCfg; paths: Paths; extract: ExtractCfg; eval: EvalCfg; limits: Limits; demo: DemoCfg = DemoCfg()

def load_config(path: str = "config.yaml") -> Config:
    data = yaml.safe_load(Path(path).read_text())
    if v := os.getenv("LLM_BASE_URL"): data["llm"]["base_url"] = v
    if v := os.getenv("LLM_MODEL"): data["llm"]["model"] = v
    return Config.model_validate(data)
