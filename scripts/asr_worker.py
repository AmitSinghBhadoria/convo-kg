"""ASR + diarization worker — runs in .venv-asr (NOT the main env).
Usage: python scripts/asr_worker.py <clip> <dev|final>
Reads data/work/<clip>.clean.wav, writes data/work/<clip>.transcript.json (Transcript schema).
HF_TOKEN is passed in the environment by the parent (src/diarize_asr.py).
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # import src.asr_merge
sys.path.insert(0, str(HERE))          # import sibling asr_worker_final (Task 8)
from src.asr_merge import merge_words_to_speakers
WORK = Path(__file__).resolve().parent.parent / "data" / "work"

def _diarize(wav):
    from pyannote.audio import Pipeline
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=os.environ["HF_TOKEN"])
    diar = pipe(str(wav))
    return [(spk, float(s.start), float(s.end)) for s, _, spk in diar.itertracks(yield_label=True)]

def _asr_dev(wav):
    import mlx_whisper
    r = mlx_whisper.transcribe(str(wav), path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                               task="translate", word_timestamps=True)
    words = []
    for seg in r["segments"]:
        for w in seg.get("words", []):
            words.append({"text": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])})
    return words

def run_dev(clip, wav):
    words = _asr_dev(wav)
    turns = _diarize(wav)
    return {"clip": clip, "snr": None, "utterances": merge_words_to_speakers(words, turns)}

def main(clip, mode):
    wav = WORK / f"{clip}.clean.wav"
    if mode == "final":
        from asr_worker_final import run_final
        data = run_final(clip, wav)
    else:
        data = run_dev(clip, wav)
    (WORK / f"{clip}.transcript.json").write_text(json.dumps(data, indent=2))
    print(f"wrote {clip}.transcript.json ({len(data['utterances'])} utterances, {mode})")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "dev")
