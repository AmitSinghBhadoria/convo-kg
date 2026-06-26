"""Pure word->speaker merge (no torch/pydantic) so it runs in main AND .venv-asr."""
def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))

def merge_words_to_speakers(words, turns):
    """words: [{'text','start','end'}]; turns: [(speaker,t0,t1)].
    Assign each word the most-overlapping turn's speaker; group consecutive same-speaker
    words into utterances. Returns [{'speaker','text','start','end','words':[...]}]."""
    out = []
    for w in words:
        # empty diarization -> default everyone to SPEAKER_00 (no crash; one merged utterance)
        best, bestov = (turns[0][0] if turns else "SPEAKER_00"), -1.0
        for spk, t0, t1 in turns:
            ov = _overlap(w["start"], w["end"], t0, t1)
            if ov > bestov:
                best, bestov = spk, ov
        wd = {**w, "speaker": best}
        if out and out[-1]["speaker"] == best:
            out[-1]["words"].append(wd); out[-1]["text"] += " " + wd["text"]; out[-1]["end"] = wd["end"]
        else:
            out.append({"speaker": best, "text": wd["text"], "start": wd["start"],
                        "end": wd["end"], "words": [wd]})
    return out
