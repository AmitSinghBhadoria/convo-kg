#!/usr/bin/env python3
"""
add_noise.py — mix background noise into a speech clip at controlled SNR levels.

Why SNR: signal-to-noise ratio (dB) is the standard, reproducible way to dial how
loud the noise is relative to speech. Higher dB = cleaner; 0 dB = noise as loud as
speech. Generating several levels lets you show an accuracy-vs-noise curve.

Usage:
  # with a real noise clip (recommended — record cafe/traffic on your phone):
  python add_noise.py --speech sample.wav --noise cafe.wav --snr 20 10 5 0

  # quick fallback with synthetic noise (no noise file handy):
  python add_noise.py --speech sample.wav --noise-type pink --snr 15 5

Outputs one 16 kHz mono WAV per SNR level into ./noisy/ (16k mono = ASR-ready).
For .mp3/.m4a inputs, install librosa (pip install librosa) or convert to wav first.
"""
import argparse, os
import numpy as np
import soundfile as sf


def load_audio(path, sr=16000):
    """Load to mono float32 at target sr. Prefers librosa (more formats), falls back to soundfile."""
    try:
        import librosa
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float32)
    except ImportError:
        y, file_sr = sf.read(path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if file_sr != sr:
            n = int(round(len(y) * sr / file_sr))
            y = np.interp(np.linspace(0, len(y), n, endpoint=False),
                          np.arange(len(y)), y).astype(np.float32)
        return y


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def make_noise(kind, length, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "white":
        return rng.standard_normal(length).astype(np.float32)
    if kind == "pink":  # 1/f noise — closer to real ambient hum than white
        white = rng.standard_normal(length)
        f = np.fft.rfftfreq(length)
        f[0] = f[1] if len(f) > 1 else 1.0
        spec = np.fft.rfft(white) / np.sqrt(f)
        pink = np.fft.irfft(spec, n=length)
        return (pink / (np.max(np.abs(pink)) + 1e-9)).astype(np.float32)
    raise ValueError(kind)


def fit_noise(noise, length, seed=0):
    """Tile noise if shorter than speech; crop at a random offset for variety."""
    if len(noise) < length:
        noise = np.tile(noise, int(np.ceil(length / len(noise))))
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, max(1, len(noise) - length + 1)))
    return noise[start:start + length]


def mix(speech, noise, snr_db):
    target_noise_rms = rms(speech) / (10 ** (snr_db / 20.0))
    noise = noise * (target_noise_rms / rms(noise))
    out = speech + noise
    peak = np.max(np.abs(out))
    if peak > 0.97:                       # prevent clipping; ratio (SNR) is preserved
        out = out * (0.97 / peak)
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech", required=True, help="path to clean speech clip")
    ap.add_argument("--noise", help="path to a real noise clip (recommended)")
    ap.add_argument("--noise-type", default="pink", choices=["white", "pink"],
                    help="synthetic noise if --noise not given")
    ap.add_argument("--snr", type=float, nargs="+", default=[20, 10, 5, 0],
                    help="SNR levels in dB (space-separated)")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--out", default="noisy")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    speech = load_audio(args.speech, args.sr)
    noise_src = load_audio(args.noise, args.sr) if args.noise else None
    base = os.path.splitext(os.path.basename(args.speech))[0]

    for snr in args.snr:
        n = noise_src.copy() if noise_src is not None else make_noise(args.noise_type, len(speech), seed=int(snr))
        n = fit_noise(n, len(speech), seed=int(snr))
        out = mix(speech, n, snr)
        path = os.path.join(args.out, f"{base}_snr{int(snr)}dB.wav")
        sf.write(path, out, args.sr)
        print(f"wrote {path}  (SNR {snr:g} dB)")


if __name__ == "__main__":
    main()
