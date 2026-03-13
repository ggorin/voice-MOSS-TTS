"""
Rate reference audio quality for voice cloning suitability.

Scores each candidate on metrics that matter for Qwen3-TTS ICL cloning:
  - SNR (signal-to-noise ratio)
  - Speech ratio (% of clip that's speech vs silence)
  - Speaker consistency (detect multiple speakers via energy variance)
  - Clipping (peaks hitting max amplitude)
  - Bandwidth (spectral energy distribution)
  - Overall score (weighted composite, 0-100)

Usage:
  cd /Users/gregorygorin/Projects/voice/mlx-tts
  source .venv/bin/activate

  # Rate a single file
  python ../MOSS-TTS/tools/rate_reference.py celebrities/audio/reference_james_earl_jones_candidate8.wav

  # Rate all candidates for a celebrity
  python ../MOSS-TTS/tools/rate_reference.py --celebrity james_earl_jones

  # Rate all candidates for all celebrities
  python ../MOSS-TTS/tools/rate_reference.py --all

  # Auto-promote best candidate per celebrity
  python ../MOSS-TTS/tools/rate_reference.py --all --auto-promote
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = SCRIPT_DIR / "celebrities" / "audio"
SAMPLE_RATE = 24000

# Score weights (must sum to 1.0)
WEIGHTS = {
    "snr": 0.30,
    "speech_ratio": 0.25,
    "consistency": 0.15,
    "clipping": 0.15,
    "bandwidth": 0.15,
}


def load_audio(path):
    """Load audio file, convert to mono float32 at 24kHz."""
    audio, sr = sf.read(str(path), dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        # Simple resample via scipy
        from scipy.signal import resample
        new_len = int(len(audio) * SAMPLE_RATE / sr)
        audio = resample(audio, new_len).astype(np.float32)
    return audio


def score_snr(audio):
    """Estimate SNR using VAD to separate speech from noise.
    Returns (snr_db, score_0_100)."""
    try:
        import torch
        from silero_vad import load_silero_vad, get_speech_timestamps

        model = load_silero_vad()
        wav_tensor = torch.tensor(audio).unsqueeze(0)

        # silero expects 16kHz
        if SAMPLE_RATE != 16000:
            wav_16k = torch.nn.functional.interpolate(
                wav_tensor.unsqueeze(0), scale_factor=16000/SAMPLE_RATE, mode="linear"
            ).squeeze()
        else:
            wav_16k = wav_tensor.squeeze()

        timestamps = get_speech_timestamps(wav_16k, model, sampling_rate=16000)

        if not timestamps:
            return 0, 0

        # Build speech/noise masks at original sample rate
        scale = SAMPLE_RATE / 16000
        speech_mask = np.zeros(len(audio), dtype=bool)
        for ts in timestamps:
            start = int(ts["start"] * scale)
            end = int(ts["end"] * scale)
            speech_mask[start:end] = True

        speech = audio[speech_mask]
        noise = audio[~speech_mask]

        if len(noise) < 100 or len(speech) < 100:
            return 40, 80  # Mostly speech, assume decent

        speech_power = np.mean(speech ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power < 1e-10:
            snr_db = 60
        else:
            snr_db = 10 * np.log10(speech_power / noise_power)

        # Score: 20dB=50, 30dB=75, 40dB+=100
        score = np.clip((snr_db - 10) * (100 / 30), 0, 100)
        return round(snr_db, 1), round(score)

    except Exception:
        # Fallback: simple RMS-based estimate
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-6:
            return 0, 0
        # Use bottom 10% as noise floor estimate
        frame_size = 1024
        n_frames = len(audio) // frame_size
        if n_frames < 2:
            return 20, 50
        frame_rms = np.array([
            np.sqrt(np.mean(audio[i*frame_size:(i+1)*frame_size] ** 2))
            for i in range(n_frames)
        ])
        noise_floor = np.percentile(frame_rms, 10)
        signal_level = np.percentile(frame_rms, 90)
        if noise_floor < 1e-10:
            snr_db = 50
        else:
            snr_db = 20 * np.log10(signal_level / noise_floor)
        score = np.clip((snr_db - 10) * (100 / 30), 0, 100)
        return round(snr_db, 1), round(score)


def score_speech_ratio(audio):
    """Measure what % of the clip is active speech.
    Returns (ratio, score_0_100)."""
    try:
        import torch
        from silero_vad import load_silero_vad, get_speech_timestamps

        model = load_silero_vad()
        wav_tensor = torch.tensor(audio)

        if SAMPLE_RATE != 16000:
            wav_16k = torch.nn.functional.interpolate(
                wav_tensor.unsqueeze(0).unsqueeze(0),
                scale_factor=16000/SAMPLE_RATE, mode="linear"
            ).squeeze()
        else:
            wav_16k = wav_tensor

        timestamps = get_speech_timestamps(wav_16k, model, sampling_rate=16000)

        if not timestamps:
            return 0, 0

        speech_samples = sum(ts["end"] - ts["start"] for ts in timestamps)
        total_samples = len(wav_16k)
        ratio = speech_samples / total_samples

        # Score: 0.6=50, 0.8=80, 0.9+=95
        score = np.clip(ratio * 110 - 10, 0, 100)
        return round(ratio, 2), round(score)

    except Exception:
        # Fallback: energy-based
        frame_size = 1024
        n_frames = len(audio) // frame_size
        if n_frames < 2:
            return 0.5, 50
        frame_rms = np.array([
            np.sqrt(np.mean(audio[i*frame_size:(i+1)*frame_size] ** 2))
            for i in range(n_frames)
        ])
        threshold = np.percentile(frame_rms, 20) * 2
        speech_frames = np.sum(frame_rms > threshold)
        ratio = speech_frames / n_frames
        score = np.clip(ratio * 110 - 10, 0, 100)
        return round(ratio, 2), round(score)


def score_consistency(audio):
    """Check for speaker consistency via energy variance across segments.
    Low variance = consistent single speaker. High variance = multiple speakers or music.
    Returns (cv, score_0_100)."""
    seg_len = int(SAMPLE_RATE * 1.0)  # 1-second segments
    n_segs = len(audio) // seg_len
    if n_segs < 3:
        return 0, 50

    seg_energies = []
    for i in range(n_segs):
        seg = audio[i*seg_len:(i+1)*seg_len]
        rms = np.sqrt(np.mean(seg ** 2))
        if rms > 1e-6:  # Skip silence
            seg_energies.append(rms)

    if len(seg_energies) < 2:
        return 0, 50

    energies = np.array(seg_energies)
    cv = np.std(energies) / np.mean(energies)  # Coefficient of variation

    # Score: cv < 0.3 = great (single speaker), cv > 0.8 = bad (multiple/music)
    score = np.clip((1 - cv) * 120 - 10, 0, 100)
    return round(cv, 3), round(score)


def score_clipping(audio):
    """Detect clipping (samples near +-1.0).
    Returns (clip_ratio, score_0_100)."""
    threshold = 0.99
    clipped = np.sum(np.abs(audio) > threshold)
    ratio = clipped / len(audio)

    # Score: 0% clipping = 100, 0.1% = 80, 1%+ = 0
    score = np.clip((1 - ratio * 1000) * 100, 0, 100)
    return round(ratio * 100, 3), round(score)


def score_bandwidth(audio):
    """Check spectral energy distribution. Good reference = energy up to 10kHz+.
    Returns (bandwidth_khz, score_0_100)."""
    from scipy.fft import rfft, rfftfreq

    spectrum = np.abs(rfft(audio))
    freqs = rfftfreq(len(audio), 1.0/SAMPLE_RATE)

    # Find frequency where 95% of energy is below
    cumulative = np.cumsum(spectrum ** 2)
    total = cumulative[-1]
    if total < 1e-10:
        return 0, 0

    idx_95 = np.searchsorted(cumulative, total * 0.95)
    bw_hz = freqs[min(idx_95, len(freqs)-1)]
    bw_khz = bw_hz / 1000

    # Score: 4kHz=40 (telephone), 8kHz=70, 10kHz+=90
    score = np.clip(bw_khz * 10, 0, 100)
    return round(bw_khz, 1), round(score)


def rate_file(path):
    """Rate a single audio file. Returns dict of metrics + overall score."""
    audio = load_audio(path)
    duration = len(audio) / SAMPLE_RATE

    snr_db, snr_score = score_snr(audio)
    speech_ratio, speech_score = score_speech_ratio(audio)
    cv, consistency_score = score_consistency(audio)
    clip_pct, clip_score = score_clipping(audio)
    bw_khz, bw_score = score_bandwidth(audio)

    overall = (
        snr_score * WEIGHTS["snr"] +
        speech_score * WEIGHTS["speech_ratio"] +
        consistency_score * WEIGHTS["consistency"] +
        clip_score * WEIGHTS["clipping"] +
        bw_score * WEIGHTS["bandwidth"]
    )

    return {
        "file": path.name,
        "duration": round(duration, 1),
        "snr_db": snr_db,
        "snr_score": snr_score,
        "speech_ratio": speech_ratio,
        "speech_score": speech_score,
        "consistency_cv": cv,
        "consistency_score": consistency_score,
        "clipping_pct": clip_pct,
        "clipping_score": clip_score,
        "bandwidth_khz": bw_khz,
        "bandwidth_score": bw_score,
        "overall": round(overall),
    }


def print_rating(r, verbose=True):
    """Pretty-print a rating."""
    grade = "A" if r["overall"] >= 85 else "B" if r["overall"] >= 70 else "C" if r["overall"] >= 55 else "D" if r["overall"] >= 40 else "F"

    if verbose:
        print(f"\n  {r['file']}  ({r['duration']}s)")
        print(f"  {'─'*50}")
        print(f"  SNR:          {r['snr_db']:>6.1f} dB   (score: {r['snr_score']})")
        print(f"  Speech ratio: {r['speech_ratio']:>6.0%}      (score: {r['speech_score']})")
        print(f"  Consistency:  {r['consistency_cv']:>6.3f} cv   (score: {r['consistency_score']})")
        print(f"  Clipping:     {r['clipping_pct']:>5.2f}%      (score: {r['clipping_score']})")
        print(f"  Bandwidth:    {r['bandwidth_khz']:>5.1f} kHz  (score: {r['bandwidth_score']})")
        print(f"  {'─'*50}")
        print(f"  OVERALL:      {r['overall']}/100  [{grade}]")
    else:
        print(f"  {r['file']:<55} {r['duration']:>5}s  SNR:{r['snr_db']:>5.1f}dB  Speech:{r['speech_ratio']:>4.0%}  BW:{r['bandwidth_khz']:>4.1f}kHz  Score:{r['overall']:>3} [{grade}]")


def main():
    parser = argparse.ArgumentParser(description="Rate reference audio quality")
    parser.add_argument("file", nargs="?", help="Audio file to rate")
    parser.add_argument("--celebrity", help="Rate all candidates for this celebrity")
    parser.add_argument("--all", action="store_true", help="Rate all candidates for all celebrities")
    parser.add_argument("--auto-promote", action="store_true", help="Auto-promote best candidate per celebrity")
    parser.add_argument("--compact", action="store_true", help="Compact output (one line per file)")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.is_absolute():
            path = SCRIPT_DIR / path
        if not path.exists():
            print(f"ERROR: File not found: {path}")
            return
        r = rate_file(path)
        print_rating(r, verbose=not args.compact)
        return

    if args.celebrity:
        slugs = [args.celebrity]
    elif args.all:
        # Find all celebrities that have candidates
        seen = set()
        for f in sorted(AUDIO_DIR.glob("reference_*_candidate*.wav")):
            slug = f.name.replace("reference_", "").rsplit("_candidate", 1)[0]
            seen.add(slug)
        slugs = sorted(seen)
    else:
        parser.print_help()
        return

    for slug in slugs:
        candidates = sorted(AUDIO_DIR.glob(f"reference_{slug}_candidate*.wav"))
        if not candidates:
            print(f"\n  {slug}: no candidates found")
            continue

        print(f"\n{'='*60}")
        print(f"  {slug} ({len(candidates)} candidates)")
        print(f"{'='*60}")

        ratings = []
        compact = args.compact or len(candidates) > 3
        for c in candidates:
            r = rate_file(c)
            print_rating(r, verbose=not compact)
            ratings.append(r)

        # Find best
        best = max(ratings, key=lambda x: x["overall"])
        print(f"\n  BEST: {best['file']} (score: {best['overall']})")

        if args.auto_promote:
            src = AUDIO_DIR / best["file"]
            dst = AUDIO_DIR / f"reference_{slug}.wav"
            shutil.copy2(src, dst)
            print(f"  Promoted to: {dst.name}")


if __name__ == "__main__":
    main()
