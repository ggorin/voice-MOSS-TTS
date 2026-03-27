# Voiceclone

Voice cloning and TTS narration on Apple Silicon using Qwen3-TTS via mlx-audio.

## Project Structure

- `generate_celebrity.py` — Unified celebrity voice clone generation (25 voices)
- `generate_adam_qwen3.py` — Adam Barrow voice clone narration (chunk-based)
- `generate_greg_qwen3.py` — Gregory Gorin voice clone narration (single-pass)
- `studio_post_process.py` — Studio mic simulation post-processing (U87, SM7B presets)
- `celebrities/` — Voice configs, monologue scripts, profiles, and reference audio
- `tools/` — Reference audio extraction (extract_reference.py, batch_extract_references.py, rate_reference.py)
- `docs/voice-cloning-process.md` — Complete voice cloning guide
- `narration_output/` — Generated audio (gitignored)

## Key Models

- **Primary:** `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` (voice cloning via ICL)
- **Legacy:** `OpenMOSS-Team/MOSS-TTS` (used by scripts in legacy/)

## Environment

```bash
cd /Users/gregorygorin/Projects/voice/voiceclone
source .venv/bin/activate
```

Uses `uv` for package management. All dependencies in pyproject.toml.

## Quick Commands

```bash
python generate_celebrity.py --list                        # list voices
python generate_celebrity.py morgan_freeman --play          # generate + play
python generate_celebrity.py --all                         # generate all
python studio_post_process.py output.wav -p u87_voiceover  # post-process
python tools/rate_reference.py --all                       # rate all references
```

## Voice Profile System

Each celebrity voice has three files:
1. `celebrities/configs/<name>.yaml` — Generation params (speed, mode, pauses)
2. `celebrities/scripts/<name>_monologue.yaml` — Monologue text
3. `celebrities/profiles/<name>-voice-profile.md` — Speaking style guide

Reference audio lives in `celebrities/audio/` (gitignored).

## Conventions

- All audio: mono, 24kHz, PCM 16-bit WAV
- Reference clips: 12-15 seconds of clean solo speech
- Generation scripts resolve paths relative to `SCRIPT_DIR = Path(__file__).resolve().parent`
