"""Audio preparation helpers — thin wrappers around ffprobe/ffmpeg.

Both functions are module-level so tests can monkeypatch them via
`monkeypatch.setattr(api.audioprep, "probe_duration", ...)`.
"""
import subprocess


def probe_duration(path: str) -> float:
    """Return audio duration in seconds.

    Shells ``ffprobe``; raises ``ValueError`` if the file is not decodable
    audio or if the duration cannot be parsed.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError(f"ffprobe could not decode audio: {result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise ValueError(f"ffprobe returned non-numeric duration: {result.stdout!r}") from exc


def to_16k_mono(src: str, dst: str) -> None:
    """Re-encode *src* to 16 kHz mono WAV at *dst*.

    Shells ``ffmpeg``; raises ``RuntimeError`` on non-zero exit.
    """
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel", "error",
            "-i", src,
            "-ac", "1",
            "-ar", "16000",
            dst,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")
