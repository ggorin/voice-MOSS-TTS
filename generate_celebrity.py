"""
Unified celebrity voice clone generation using Qwen3-TTS Base ICL mode.

Supports both chunked and single-pass generation modes, configured per
celebrity via YAML configs. Replaces per-voice scripts.

Usage:
  cd /Users/gregorygorin/Projects/voice/voiceclone
  source .venv/bin/activate
  python generate_celebrity.py morgan_freeman              # generate monologue
  python generate_celebrity.py morgan_freeman --play        # generate + play
  python generate_celebrity.py --all                       # generate all 22
  python generate_celebrity.py --list                      # list available voices
  python generate_celebrity.py morgan_freeman --text "..." # custom text
"""

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from mlx_audio.tts.utils import load_model

# ─── Paths ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPT_DIR / "celebrities" / "configs"
SCRIPTS_DIR = SCRIPT_DIR / "celebrities" / "scripts"
OUTPUT_DIR = SCRIPT_DIR / "narration_output"

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
SAMPLE_RATE = 24000


def load_config(slug):
    """Load celebrity YAML config."""
    config_path = CONFIGS_DIR / f"{slug}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_script(slug):
    """Load celebrity monologue script."""
    script_path = SCRIPTS_DIR / f"{slug}_monologue.yaml"
    if not script_path.exists():
        raise FileNotFoundError(f"No script found: {script_path}")
    with open(script_path) as f:
        return yaml.safe_load(f)


def list_celebrities():
    """List all available celebrity configs."""
    configs = sorted(CONFIGS_DIR.glob("*.yaml"))
    if not configs:
        print("No celebrity configs found in celebrities/configs/")
        return
    print(f"\n{'='*60}")
    print(f"  Available Celebrity Voices ({len(configs)})")
    print(f"{'='*60}")
    for cfg_path in configs:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        slug = cfg_path.stem
        name = cfg.get("name", slug)
        speed = cfg.get("generation", {}).get("speed", "?")
        mode = cfg.get("generation", {}).get("mode", "?")
        ref_exists = (SCRIPT_DIR / cfg.get("ref_audio", "")).exists() if cfg.get("ref_audio") else False
        status = "ready" if ref_exists else "needs ref audio"
        print(f"  {slug:<30} {name:<25} speed={speed}  mode={mode}  [{status}]")
    print()


def generate_single_pass(model, config, text, output_path):
    """Generate audio in a single pass (best for voice consistency)."""
    ref_audio = str(SCRIPT_DIR / config["ref_audio"])
    ref_text = config["ref_text"]
    speed = config["generation"]["speed"]

    word_count = len(text.split())
    print(f"\n{'─'*60}")
    print(f"  Single-pass generation ({word_count} words)")
    print(f"{'─'*60}")

    t0 = time.time()
    results = list(model.generate(
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        speed=speed,
    ))
    audio = np.array(results[0].audio)
    dur = len(audio) / SAMPLE_RATE
    elapsed = time.time() - t0
    print(f"  Generated {dur:.1f}s audio in {elapsed:.1f}s (RTF: {elapsed/dur:.1f}x)")

    sf.write(str(output_path), audio, SAMPLE_RATE)
    print(f"  Saved: {output_path}")

    return audio


def generate_chunked(model, config, chunks, output_path):
    """Generate audio in chunks with pauses between them."""
    ref_audio = str(SCRIPT_DIR / config["ref_audio"])
    ref_text = config["ref_text"]
    speed = config["generation"]["speed"]
    pause_normal_ms = config["generation"].get("pause_normal_ms", 450)
    pause_short_ms = config["generation"].get("pause_short_ms", 350)

    silence_normal = np.zeros(int(SAMPLE_RATE * pause_normal_ms / 1000), dtype=np.float32)
    silence_short = np.zeros(int(SAMPLE_RATE * pause_short_ms / 1000), dtype=np.float32)

    print(f"\n{'─'*60}")
    print(f"  Chunked generation ({len(chunks)} chunks)")
    print(f"{'─'*60}")

    audio_parts = []
    t_start = time.time()

    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        pause_type = chunk.get("pause", "none")
        print(f"  [{i+1}/{len(chunks)}] {text[:65]}{'...' if len(text) > 65 else ''}")

        chunk_speed = chunk.get("speed", speed)
        t1 = time.time()
        results = list(model.generate(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            speed=chunk_speed,
        ))
        chunk_audio = np.array(results[0].audio)
        audio_parts.append(chunk_audio)
        dur = len(chunk_audio) / SAMPLE_RATE
        print(f"         {time.time()-t1:.1f}s -> {dur:.1f}s audio")

        if pause_type == "normal":
            audio_parts.append(silence_normal)
        elif pause_type == "short":
            audio_parts.append(silence_short)

    final = np.concatenate(audio_parts)
    total_dur = len(final) / SAMPLE_RATE
    elapsed = time.time() - t_start

    sf.write(str(output_path), final, SAMPLE_RATE)
    print(f"\n  Saved: {output_path}")
    print(f"  {total_dur:.1f}s audio in {elapsed:.1f}s (RTF: {elapsed/total_dur:.1f}x)")

    return final


def generate_celebrity(model, slug, custom_text=None, slide_key="monologue"):
    """Generate monologue for a single celebrity."""
    config = load_config(slug)
    name = config["name"]
    mode = config["generation"]["mode"]
    speed = config["generation"]["speed"]

    # Resolve reference audio
    ref_path = SCRIPT_DIR / config["ref_audio"]
    if not ref_path.exists():
        print(f"\n  WARNING: Reference audio not found: {ref_path}")
        print(f"  Use tools/extract_reference.py to create it first.")
        return None, None

    print(f"\n{'='*60}")
    print(f"  {name} — Qwen3-TTS Base ICL")
    print(f"  Speed: {speed} | Mode: {mode}")
    print(f"{'='*60}")

    output_path = OUTPUT_DIR / f"{slug}_{slide_key}.wav"

    if custom_text:
        # Custom text always uses single-pass
        return generate_single_pass(model, config, custom_text, output_path), output_path

    # Load monologue script
    script = load_script(slug)
    monologue = script["slides"][slide_key]
    script_mode = monologue.get("mode", mode)

    if script_mode == "single_pass":
        text = monologue["text"]
        audio = generate_single_pass(model, config, text, output_path)
    elif script_mode == "chunked":
        chunks = monologue["chunks"]
        audio = generate_chunked(model, config, chunks, output_path)
    else:
        raise ValueError(f"Unknown mode: {script_mode}")

    return audio, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate celebrity voice clone narration (Qwen3-TTS)",
    )
    parser.add_argument(
        "celebrity", nargs="?",
        help="Celebrity slug (e.g., morgan_freeman)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate all available celebrities",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available celebrity voices",
    )
    parser.add_argument(
        "--play", action="store_true",
        help="Play audio after generation",
    )
    parser.add_argument(
        "--text", type=str,
        help="Custom text to generate (single-pass mode)",
    )
    parser.add_argument(
        "--slide", type=str, default="monologue",
        help="Which slide to generate from the script (default: monologue)",
    )
    args = parser.parse_args()

    if args.list:
        list_celebrities()
        return

    if not args.celebrity and not args.all:
        parser.print_help()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model once
    print("\nLoading model...")
    t0 = time.time()
    model = load_model(MODEL_ID)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    if args.all:
        # Generate all celebrities
        configs = sorted(CONFIGS_DIR.glob("*.yaml"))
        results = []
        for cfg_path in configs:
            slug = cfg_path.stem
            try:
                audio, path = generate_celebrity(model, slug)
                if audio is not None:
                    results.append((slug, path))
            except Exception as e:
                print(f"\n  ERROR generating {slug}: {e}")

        print(f"\n{'='*60}")
        print(f"  Generated {len(results)}/{len(configs)} celebrity voices")
        for slug, path in results:
            print(f"    {slug}: {path}")
        print(f"{'='*60}")
    else:
        audio, path = generate_celebrity(model, args.celebrity, custom_text=args.text, slide_key=args.slide)

        if args.play and path:
            print(f"\nPlaying: {path}")
            subprocess.run(["afplay", str(path)])

    print("\nDone!")


if __name__ == "__main__":
    main()
