"""WhisperX final pass — runs in .venv-asr. Imported by asr_worker.py for mode='final'."""
import os
import whisperx

def run_final(clip, wav):
    device, compute = "cpu", "int8"        # M4: CPU/int8 keeps memory low
    model = whisperx.load_model("large-v3", device, compute_type=compute, task="translate")
    audio = whisperx.load_audio(str(wav))
    result = model.transcribe(audio, batch_size=8)
    amodel, meta = whisperx.load_align_model(language_code="en", device=device)
    result = whisperx.align(result["segments"], amodel, meta, audio, device, return_char_alignments=False)
    dia = whisperx.DiarizationPipeline(use_auth_token=os.environ["HF_TOKEN"], device=device)
    result = whisperx.assign_word_speakers(dia(audio), result)
    utts = []
    for seg in result["segments"]:
        spk = seg.get("speaker", "SPEAKER_00")
        words = [{"text": w["word"].strip(),
                  "start": float(w.get("start", seg["start"])),
                  "end": float(w.get("end", seg["end"])),
                  "speaker": w.get("speaker", spk)} for w in seg.get("words", [])]
        utts.append({"speaker": spk, "text": seg["text"].strip(),
                     "start": float(seg["start"]), "end": float(seg["end"]), "words": words})
    return {"clip": clip, "snr": None, "utterances": utts}
