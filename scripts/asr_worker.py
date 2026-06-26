"""ASR + diarization worker — runs in .venv-asr (NOT the main env).
Usage: python scripts/asr_worker.py <clip>
Reads data/work/<clip>.clean.wav, writes data/work/<clip>.transcript.json (Transcript schema).
HF_TOKEN is passed in the environment by the parent (src/diarize_asr.py).

ASR is mlx-whisper with task="translate" (Hinglish audio -> English text). We do
NOT use WhisperX forced-alignment: its wav2vec aligner needs text in the SAME
language as the audio, which is incompatible with translate-to-English on Hindi
audio (it drops/garbles segments). Speaker attribution is segment-level via
pyannote turns + Whisper's own word timestamps. See design_note.md.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # import src.asr_merge
from src.asr_merge import merge_words_to_speakers
WORK = Path(__file__).resolve().parent.parent / "data" / "work"

def _diarize(wav):
    from pyannote.audio import Pipeline
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=os.environ["HF_TOKEN"])
    diar = pipe(str(wav))
    return [(spk, float(s.start), float(s.end)) for s, _, spk in diar.itertracks(yield_label=True)]

def _asr(wav):
    import mlx_whisper
    r = mlx_whisper.transcribe(str(wav), path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                               task="translate", word_timestamps=True)
    words = []
    for seg in r["segments"]:
        for w in seg.get("words", []):
            words.append({"text": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])})
    return words

def run_asr(clip, wav):
    words = _asr(wav)
    turns = _diarize(wav)
    return {"clip": clip, "snr": None, "utterances": merge_words_to_speakers(words, turns)}

def main(clip):
    wav = WORK / f"{clip}.clean.wav"
    data = run_asr(clip, wav)
    (WORK / f"{clip}.transcript.json").write_text(json.dumps(data, indent=2))
    print(f"wrote {clip}.transcript.json ({len(data['utterances'])} utterances)")

if __name__ == "__main__":
    main(sys.argv[1])
