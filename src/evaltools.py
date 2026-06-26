import re
from difflib import SequenceMatcher

def normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]", "", text.lower())
    text = re.sub(r" +", " ", text)
    return text.strip()

def transcript_text(t) -> str:
    return " ".join(u.text for u in t.utterances)

def similarity(hyp: str, ref: str) -> float:
    h, r = normalize(hyp), normalize(ref)
    seq = SequenceMatcher(None, h, r).ratio()
    hs, rs = set(h.split()), set(r.split())
    jacc = len(hs & rs) / len(hs | rs) if (hs | rs) else 0.0
    return round(0.5 * seq + 0.5 * jacc, 4)
