# Voiceclone

Voice cloning and TTS narration on Apple Silicon. Uses Qwen3-TTS (ICL mode) for zero-shot voice cloning with 25+ voice profiles.

## Setup

```bash
cd /Users/gregorygorin/Projects/voice/voiceclone
uv venv && source .venv/bin/activate
uv sync
```

## Usage

```bash
# List available voices
python generate_celebrity.py --list

# Generate a monologue in Morgan Freeman's voice
python generate_celebrity.py morgan_freeman --play

# Generate all 25 celebrity monologues
python generate_celebrity.py --all

# Custom text with any voice
python generate_celebrity.py david_attenborough --text "Your text here."

# Post-process with studio mic simulation
python studio_post_process.py narration_output/morgan_freeman_monologue.wav -p u87_voiceover
```

## Voice Profiles

Each voice has a config (`celebrities/configs/`), monologue script (`celebrities/scripts/`), and speaking style profile (`celebrities/profiles/`). See `docs/voice-cloning-process.md` for the full guide.

## Models

| Model | Use | Engine |
|-------|-----|--------|
| Qwen3-TTS Base 1.7B | Voice cloning (primary) | mlx-audio |
| MOSS-TTS | Legacy narration | transformers |

## Project Layout

```
generate_celebrity.py       # Main: celebrity voice generation
generate_adam_qwen3.py      # Adam Barrow narration (chunked)
generate_greg_qwen3.py      # Gregory Gorin narration (single-pass)
studio_post_process.py      # Audio post-processing
celebrities/                # Voice configs, scripts, profiles, audio
tools/                      # Reference audio extraction + rating
docs/                       # Voice cloning process guide
legacy/                     # MOSS-TTS scripts (not actively used)
vendor/                     # Upstream model code (MOSS-TTS, AP-BWE)
```
