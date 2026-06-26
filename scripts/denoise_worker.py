"""DeepFilterNet denoise worker — runs in the ISOLATED .venv-denoise (torchaudio<2.1),
which the main env cannot host (DeepFilterNet needs torchaudio 1.x APIs; the ASR stack needs 2.x).
Invoked as a subprocess by src/enhance.py.
Usage: python scripts/denoise_worker.py <in.wav> <out_16k_mono.wav>
"""
import sys, warnings
warnings.filterwarnings("ignore")
import soundfile as sf
import torchaudio
from df.enhance import init_df, enhance, load_audio

def main(src: str, dst: str) -> None:
    model, df_state, _ = init_df()                  # loads DeepFilterNet3 (cached after first run)
    audio, _ = load_audio(src, sr=df_state.sr())    # 48 kHz
    out = enhance(model, df_state, audio)           # torch tensor @ 48 kHz
    if out.dim() == 1:
        out = out.unsqueeze(0)
    out16 = torchaudio.functional.resample(out, df_state.sr(), 16000).squeeze(0).cpu().numpy()
    sf.write(dst, out16, 16000)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
