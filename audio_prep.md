# Audio Source & Clip Prep — Atyx Prototype

Context for Claude Code: how to source and prepare the audio for the conversational
knowledge-graph prototype. (Pipeline: denoise → diarization → Hinglish→English ASR →
fact extraction → knowledge graph → single-hop Q&A, on a local open-weight LLM.)

## Source video
**"The Journey of Four Friends Building a Successful Wealth Management Business in India"**
YouTube: https://www.youtube.com/watch?v=-vHtmnsM-Tk

Multi-speaker, Hinglish, wealth-management domain (on-theme for Atyx). Use the extracted
facts (people, firm, roles, numbers/AUM, milestones, decisions) to shape the ontology —
swap the trip-planning template for this domain.

## 1. Download audio (highest quality source; we downsample anyway)
```bash
yt-dlp -x --audio-format wav "https://www.youtube.com/watch?v=-vHtmnsM-Tk"
```

## 2. Normalize to ASR-ready format (mono, 16 kHz)
```bash
ffmpeg -i input.* -ac 1 -ar 16000 full.wav
```

## 3. Cut two clips — optimize for length, not file size (16k mono ≈ 2 MB/min)
- `data/raw/sample.wav` — **3–5 min demo clip.** Pick the most **fact-dense** window
  (concrete names, numbers, dates, decisions; multiple speakers), not necessarily the start.
- `data/raw/dev.wav` — **30–60 sec dev slice** from within that window, for fast iteration.

```bash
ffmpeg -i full.wav -ss 00:02:00 -t 00:04:00 data/raw/sample.wav   # adjust window
ffmpeg -i full.wav -ss 00:02:30 -t 00:00:45 data/raw/dev.wav
```

## 4. Workflow
- Tune **denoise → diarization → Hinglish→English ASR** on `dev.wav` (seconds per loop).
- Only run the full `sample.wav` once a stage works.
- **Do NOT** process the whole 30–60 min interview — diarization/ASR errors compound,
  ground-truth labeling becomes a slog, and iteration crawls.
- Note in `design_note.md` that it scales to longer audio without demoing on a full episode.

## 5. Verify by ear before committing (~30s)
- Genuinely **Hinglish** (real code-mixing, not pure English or pure Hindi).
- **Fact-dense** (concrete names/numbers/dates/decisions, not just anecdotal).
- **Multiple distinct speakers** in the chosen segment.

## 6. Add noise at controlled SNR (after clipping)
```bash
python scripts/add_noise.py --speech data/raw/sample.wav --noise cafe.wav --snr 20 10 5 0
```
Prefer a real ambient/babble noise clip over synthetic. Build a hand-labeled ground-truth
answer key for `sample.wav` so Q&A correctness is verifiable in the demo.

---
Attribute the source URL in `design_note.md` (public clip used as test input).
