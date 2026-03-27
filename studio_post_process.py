"""
Studio microphone simulation for TTS output.

Transforms flat TTS audio into something that sounds like it was recorded
through a Neumann U87 or Shure SM7B in a treated vocal booth.

Full pipeline:
  1. AP-BWE: 24kHz -> 48kHz bandwidth extension (optional)
  2. Pre-EQ: Boost presence/sparkle for harmonic exciter
  3. Harmonic exciter: Generate HF content from mid-range via waveshaping
  4. Air noise: Envelope-modulated noise in 10-22kHz for open mic feel
  5. Studio chain: highpass -> gate -> EQ -> compression -> saturation -> reverb -> limiter

Usage:
  python studio_post_process.py narration_output/adam_barrow_slide1_hook.wav
  python studio_post_process.py input.wav -p sm7b_podcast -o output.wav
  python studio_post_process.py input.wav --warmth 1.5 --saturation 3.0
  python studio_post_process.py input.wav --no-exciter  # skip harmonic exciter
  python studio_post_process.py input.wav --no-upsample  # skip AP-BWE upsampling
"""

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
from pedalboard import (
    Compressor,
    Distortion,
    Gain,
    HighpassFilter,
    HighShelfFilter,
    Limiter,
    LowShelfFilter,
    NoiseGate,
    Pedalboard,
    PeakFilter,
    Reverb,
)
from scipy.signal import butter, sosfilt

# ─── Presets ─────────────────────────────────────────────────────────────
PRESETS = {
    "u87_voiceover": dict(
        mic_style="u87",
        warmth=1.0,
        presence=1.0,
        compression=1.0,
        saturation_db=2.0,
        air_db=1.5,
        room_size=0.03,
    ),
    "sm7b_podcast": dict(
        mic_style="sm7b",
        warmth=1.2,
        presence=0.9,
        compression=1.2,
        saturation_db=3.0,
        air_db=1.0,
        room_size=0.02,
    ),
    "broadcast_clean": dict(
        mic_style="sm7b",
        warmth=0.8,
        presence=1.1,
        compression=1.5,
        saturation_db=1.0,
        air_db=0.5,
        room_size=0.0,
    ),
    "warm_intimate": dict(
        mic_style="u87",
        warmth=1.5,
        presence=0.7,
        compression=0.8,
        saturation_db=4.0,
        air_db=2.0,
        room_size=0.05,
    ),
}


# ─── Harmonic Exciter ────────────────────────────────────────────────────

def harmonic_exciter(
    audio: np.ndarray,
    sample_rate: int,
    drive: float = 0.6,
    mix: float = 0.30,
    source_lo: int = 1500,
    source_hi: int = 10000,
    harmonics_hp: int = 5000,
) -> np.ndarray:
    """Generate high-frequency harmonics from mid-range via nonlinear waveshaping.

    Isolates source band, applies tanh + cubic saturation to create 2nd/3rd
    harmonics, then highpasses to keep only new HF content and mixes back in.
    Similar to Aphex Aural Exciter / BBE Sonic Maximizer.
    """
    sos_lo = butter(4, source_lo, btype="high", fs=sample_rate, output="sos")
    sos_hi = butter(4, source_hi, btype="low", fs=sample_rate, output="sos")
    band = sosfilt(sos_lo, audio)
    band = sosfilt(sos_hi, band)

    peak = np.max(np.abs(band)) + 1e-10
    band_norm = band / peak

    # Odd harmonics (3rd, 5th) from tanh + even harmonics (2nd) from |x|*x
    sat_odd = np.tanh(band_norm * (1 + drive * 12))
    sat_even = band_norm * np.abs(band_norm)
    saturated = 0.7 * sat_odd + 0.3 * sat_even

    # Keep only newly generated harmonics
    sos_hp = butter(6, harmonics_hp, btype="high", fs=sample_rate, output="sos")
    harmonics = sosfilt(sos_hp, saturated) * peak

    return audio + harmonics * mix


def air_noise(
    audio: np.ndarray,
    sample_rate: int,
    level_db: float = -32,
) -> np.ndarray:
    """Add envelope-modulated noise in the air band (10-22kHz).

    Simulates the natural high-frequency content present in studio
    microphone recordings. Noise follows the speech envelope so it
    only appears where there's signal.
    """
    n = len(audio)
    noise = np.random.randn(n).astype(np.float32)

    # Bandpass 10-22kHz
    sos_lo = butter(4, 10000, btype="high", fs=sample_rate, output="sos")
    sos_hi = butter(4, 22000, btype="low", fs=sample_rate, output="sos")
    noise = sosfilt(sos_lo, noise)
    noise = sosfilt(sos_hi, noise)

    # Envelope follower
    envelope = np.abs(audio)
    sos_env = butter(2, 20, btype="low", fs=sample_rate, output="sos")
    envelope = sosfilt(sos_env, envelope)
    envelope = np.clip(envelope / (np.percentile(envelope, 95) + 1e-10), 0, 1)

    level = 10 ** (level_db / 20)
    return audio + noise * level * envelope


# ─── AP-BWE Upsampling ──────────────────────────────────────────────────

def upsample_apbwe(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Upsample 24kHz audio to 48kHz using AP-BWE bandwidth extension.

    Returns (audio_48k, 48000). Falls back to torchaudio resample if
    AP-BWE checkpoint is not available.
    """
    import torch
    import torchaudio.functional as aF

    ckpt_path = os.path.join(
        os.path.dirname(__file__), "tools", "AP-BWE", "checkpoints", "24kto48k", "g_24kto48k"
    )
    config_path = os.path.join(os.path.dirname(ckpt_path), "config.json")

    if not os.path.isfile(ckpt_path):
        print("  AP-BWE checkpoint not found, using torchaudio resample")
        audio_t = torch.from_numpy(audio).unsqueeze(0)
        audio_48k = aF.resample(audio_t, orig_freq=sample_rate, new_freq=48000)
        return audio_48k.squeeze().numpy(), 48000

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools", "AP-BWE"))
    from env import AttrDict
    from datasets.dataset import amp_pha_stft, amp_pha_istft
    from models.model import APNet_BWE_Model

    with open(config_path) as f:
        h = AttrDict(json.loads(f.read()))

    device = torch.device("cpu")
    model = APNet_BWE_Model(h).to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict["generator"])
    model.eval()

    audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)
    audio_hr = aF.resample(audio_t, orig_freq=sample_rate, new_freq=h.hr_sampling_rate)
    audio_lr = aF.resample(audio_t, orig_freq=sample_rate, new_freq=h.lr_sampling_rate)
    audio_lr = aF.resample(audio_lr, orig_freq=h.lr_sampling_rate, new_freq=h.hr_sampling_rate)
    audio_lr = audio_lr[:, : audio_hr.size(1)]

    with torch.no_grad():
        amp_nb, pha_nb, com_nb = amp_pha_stft(audio_lr, h.n_fft, h.hop_size, h.win_size)
        amp_wb, pha_wb, com_wb = model(amp_nb, pha_nb)
        audio_out = amp_pha_istft(amp_wb, pha_wb, h.n_fft, h.hop_size, h.win_size)

    return audio_out.squeeze().cpu().numpy(), h.hr_sampling_rate


# ─── Studio Chain ────────────────────────────────────────────────────────

def studio_process(
    audio: np.ndarray,
    sample_rate: int = 24000,
    mic_style: str = "u87",
    warmth: float = 1.0,
    presence: float = 1.0,
    compression: float = 1.0,
    saturation_db: float = 2.0,
    air_db: float = 1.5,
    room_size: float = 0.03,
) -> np.ndarray:
    """Apply studio voiceover processing chain to audio."""
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)

    effects = []

    # 1. High-pass: remove rumble / DC
    effects.append(HighpassFilter(cutoff_frequency_hz=75))

    # 2. Noise gate: clean silence gaps
    effects.append(NoiseGate(threshold_db=-35, ratio=2.0, release_ms=200))

    # 3. EQ: warmth (proximity effect simulation)
    effects.append(
        LowShelfFilter(cutoff_frequency_hz=200, gain_db=3.0 * warmth, q=0.7)
    )

    # 4. EQ: cut mud at 300Hz
    effects.append(PeakFilter(cutoff_frequency_hz=300, gain_db=-2.0, q=1.0))

    # 5. EQ: presence peak (mic-dependent)
    if mic_style == "u87":
        effects.append(
            PeakFilter(cutoff_frequency_hz=4000, gain_db=2.5 * presence, q=1.0)
        )
        effects.append(
            PeakFilter(cutoff_frequency_hz=9000, gain_db=2.0 * presence, q=1.5)
        )
    elif mic_style == "sm7b":
        effects.append(
            PeakFilter(cutoff_frequency_hz=3500, gain_db=1.5 * presence, q=0.7)
        )
        effects.append(
            PeakFilter(cutoff_frequency_hz=7500, gain_db=1.0 * presence, q=1.0)
        )

    # 6. EQ: air shelf
    effects.append(HighShelfFilter(cutoff_frequency_hz=10000, gain_db=air_db, q=0.7))

    # 7. Compression
    effects.append(
        Compressor(
            threshold_db=-18.0 * compression,
            ratio=4.0,
            attack_ms=15.0,
            release_ms=150.0,
        )
    )

    # 8. Saturation (analog warmth via soft clipping)
    if saturation_db > 0:
        effects.append(Distortion(drive_db=saturation_db))

    # 9. Room ambience (tiny booth)
    if room_size > 0:
        effects.append(Reverb(room_size=room_size, wet_level=0.08, dry_level=1.0))

    # 10. Limiter + headroom
    effects.append(Limiter(threshold_db=-1.0, release_ms=100.0))
    effects.append(Gain(gain_db=-0.5))

    board = Pedalboard(effects)
    processed = board(audio, sample_rate)
    return processed.reshape(-1)


def full_pipeline(
    audio: np.ndarray,
    sample_rate: int = 24000,
    upsample: bool = True,
    exciter: bool = True,
    preset: str = "u87_voiceover",
    **kwargs,
) -> tuple[np.ndarray, int]:
    """Full studio post-processing pipeline for TTS output.

    Returns (processed_audio, output_sample_rate).
    """
    params = dict(PRESETS[preset])
    params.update(kwargs)

    # Stage 1: Upsample to 48kHz
    if upsample and sample_rate < 48000:
        print("  Stage 1: AP-BWE upsample 24kHz -> 48kHz...")
        audio, sample_rate = upsample_apbwe(audio, sample_rate)

    if exciter and sample_rate >= 48000:
        # Stage 2: Pre-EQ to boost presence for exciter
        print("  Stage 2: Pre-EQ + harmonic exciter...")
        pre_eq = Pedalboard([
            PeakFilter(cutoff_frequency_hz=3000, gain_db=4.0, q=0.8),
            PeakFilter(cutoff_frequency_hz=5000, gain_db=3.0, q=1.0),
            HighShelfFilter(cutoff_frequency_hz=8000, gain_db=3.0, q=0.7),
        ])
        audio = pre_eq(audio.reshape(1, -1), sample_rate).reshape(-1)

        # Stage 3: Harmonic exciter
        audio = harmonic_exciter(audio, sample_rate)

        # Stage 4: Air noise
        print("  Stage 3: Air noise...")
        audio = air_noise(audio, sample_rate)

    # Stage 5: Studio chain
    print("  Stage 4: Studio chain (%s)..." % preset)
    audio = studio_process(audio, sample_rate, **params)

    return audio, sample_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Studio mic post-processor for TTS")
    parser.add_argument("input", help="Input WAV file")
    parser.add_argument("-o", "--output", help="Output WAV file")
    parser.add_argument(
        "-p",
        "--preset",
        choices=list(PRESETS.keys()),
        default="u87_voiceover",
        help="Processing preset (default: u87_voiceover)",
    )
    parser.add_argument("--warmth", type=float, help="Warmth multiplier (0-2)")
    parser.add_argument("--presence", type=float, help="Presence multiplier (0-2)")
    parser.add_argument("--saturation", type=float, help="Saturation in dB")
    parser.add_argument("--air", type=float, help="Air shelf boost in dB")
    parser.add_argument(
        "--no-exciter", action="store_true", help="Skip harmonic exciter"
    )
    parser.add_argument(
        "--no-upsample", action="store_true", help="Skip AP-BWE upsampling"
    )
    args = parser.parse_args()

    audio, sr = sf.read(args.input, dtype="float32")
    print("Loaded: %s (%.1fs @ %dHz)" % (args.input, len(audio) / sr, sr))

    overrides = {}
    if args.warmth is not None:
        overrides["warmth"] = args.warmth
    if args.presence is not None:
        overrides["presence"] = args.presence
    if args.saturation is not None:
        overrides["saturation_db"] = args.saturation
    if args.air is not None:
        overrides["air_db"] = args.air

    print("Preset: %s" % args.preset)
    processed, out_sr = full_pipeline(
        audio,
        sr,
        upsample=not args.no_upsample,
        exciter=not args.no_exciter,
        preset=args.preset,
        **overrides,
    )

    out_path = args.output or args.input.replace(".wav", "_studio.wav")
    sf.write(out_path, processed, out_sr)
    print("Saved: %s (%.1fs @ %dHz)" % (out_path, len(processed) / out_sr, out_sr))
