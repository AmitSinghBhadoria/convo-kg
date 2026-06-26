from pydantic import BaseModel, Field

class Word(BaseModel):
    text: str
    start: float
    end: float
    speaker: str

class Utterance(BaseModel):
    speaker: str
    text: str
    start: float
    end: float
    words: list[Word] = Field(default_factory=list)

class Transcript(BaseModel):
    clip: str
    snr: str | None = None
    utterances: list[Utterance] = Field(default_factory=list)

class Entity(BaseModel):
    id: str               # stable dedupe key, e.g. "entity:pms"
    label: str            # fixed backbone: Speaker|Statement|Entity|Claim|Attribute
    type: str             # induced open vocabulary
    name: str
    attrs: dict = Field(default_factory=dict)

class Fact(BaseModel):
    subject_id: str
    relation: str         # induced open vocabulary
    object_id: str
    statement: str        # MANDATORY source grounding
    speaker: str
    confidence: float

class FactSet(BaseModel):
    clip: str
    entities: list[Entity] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
