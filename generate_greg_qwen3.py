"""
Generate Gregory Gorin voice clone narration using Qwen3-TTS Base ICL mode.

Single-pass generation per slide for consistent voice, with Gregory-style
tics and technical delivery patterns.

Usage:
  cd /Users/gregorygorin/Projects/voice/mlx-tts
  source .venv/bin/activate
  python ../MOSS-TTS/generate_greg_qwen3.py                    # all slides
  python ../MOSS-TTS/generate_greg_qwen3.py --slide 1          # slide 1 only
  python ../MOSS-TTS/generate_greg_qwen3.py --slide 1 --play   # generate + play
"""

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model

# ─── Reference Audio ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REF_AUDIO = str(SCRIPT_DIR / "assets" / "audio" / "reference_gregory_gorin.wav")
REF_TEXT = (
    "I think it's just kind of workflow. How much do we want to live in this "
    "app versus Pipedrive? Is this just a contact to find the contacts? And "
    "then we put them over to Pipedrive. I know Apollo onto itself has some "
    "great functionality."
)
OUTPUT_DIR = SCRIPT_DIR / "narration_output"

# ─── Generation Settings ────────────────────────────────────────────────
MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
SPEED = 0.88
SAMPLE_RATE = 24000

# ─── Slide Scripts (single-pass per slide for voice consistency) ────────
# Each slide is one continuous text block. The model generates it in one pass
# to maintain consistent voice throughout. Natural pauses come from punctuation.
#
# Gregory's voice patterns:
#   - Opens with "Yeah, so" / "Okay, so" / "I mean"
#   - Heavy use of "so" as transition between every idea
#   - "I think," "you know," "kind of," "actually" as natural hedges
#   - Self-corrections: "... I mean," restarts, mid-thought pivots
#   - Step-by-step logical flow when explaining
#   - Specific metrics with context ("that's huge," "that's an issue")
#   - No salesy language — technical, direct, analytical

SLIDES = {
    "slide1_hook": (
        "Yeah, so here's what I wanted to show you. "
        "You've got three hundred and six reviews at four point seven stars. "
        "I mean, that's actually really solid. "
        "But the thing is, Fishkill Cannabis has basically one more review than you "
        "and they're still outranking you. "
        "And when you look at why, it's the website. "
        "They've got meta descriptions, they've got schema markup, "
        "you know, proper on-page S.E.O. stuff. "
        "So you've got the reviews, you've got the reputation, "
        "but the site's sitting at forty two out of a hundred. "
        "And I think the good news is, a lot of this is actually kind of "
        "low-hanging fruit. Like, it's fixable pretty fast."
    ),
    "slide2_audit": (
        "Okay, so let me walk you through what we found. "
        "Your Google Business Profile scores ninety out of a hundred. "
        "So that's great. And you're already ranking number one for dispensary Fishkill. "
        "But here's where it gets kind of interesting. "
        "You've got zero meta descriptions across all fifteen pages. "
        "No H1 tag on the homepage. "
        "So Google basically can't tell what your main heading is. "
        "No schema markup. And your competitors, they have it. "
        "There's like eleven broken links, which is definitely hurting your user experience. "
        "And then the homepage itself only has about two hundred fifty words, "
        "and Google sees that as thin content. "
        "So I mean, there's a lot we can do here. "
        "And most of it's not that complicated to fix."
    ),
    "slide3_plan": (
        "Yeah, so here's the ninety day plan. "
        "We've mapped out sixteen specific moves. "
        "And I think we're gonna see the score go from like forty two "
        "up to somewhere between eighty and ninety. Which is a huge jump. "
        "So phase one is basically stop the bleeding. This week. "
        "We add meta descriptions to all fifteen pages. "
        "Get the H1 headings in. Implement schema markup. "
        "And fix those eleven broken links. "
        "Phase two, we build the foundation. "
        "So we fix the N.A.P. inconsistencies, get you a Leafly profile, "
        "and build out a proper location landing page. "
        "And then phase three is where we really kind of go after the market. "
        "We expand the content, create landing pages for Beacon, Wappingers, Newburgh, "
        "and launch a monthly blog to keep things moving. "
        "So yeah, I think the results on this are gonna be pretty significant."
    ),
    "slide4_cta": (
        "Okay so, I mean, getting you guys into that number one spot is very doable. "
        "All you gotta do is reply to this email and say hey, I'm interested. That's it. "
        "No commitment, no pressure. "
        "I think you'll see pretty quickly that this makes sense. "
        "We'll hop on a fifteen minute call, walk through the top five quick wins together. "
        "And then from there, we handle all the technical stuff. "
        "You keep doing what you're doing, which is running a great dispensary. "
        "And actually, before we even get on that call, "
        "we're gonna deliver three things. "
        "A custom Google Business Profile description. "
        "Meta description copy for your top pages. "
        "And a full list of every N.A.P. mismatch with the exact corrections. "
        "So yeah. You know, we want to earn your trust before we even start. "
        "Let us show you what we can do."
    ),
}


def generate_slide(model, slide_name, text):
    """Generate a single slide in one pass for voice consistency."""
    print(f"\n{'─'*60}")
    print(f"  {slide_name} ({len(text.split())} words)")
    print(f"{'─'*60}")

    t0 = time.time()
    results = list(model.generate(
        text=text,
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
        speed=SPEED,
    ))
    audio = np.array(results[0].audio)
    dur = len(audio) / SAMPLE_RATE
    elapsed = time.time() - t0
    print(f"  Generated {dur:.1f}s audio in {elapsed:.1f}s (RTF: {elapsed/dur:.1f}x)")

    out_path = OUTPUT_DIR / f"qwen3_greg_{slide_name}.wav"
    sf.write(str(out_path), audio, SAMPLE_RATE)
    print(f"  Saved: {out_path}")

    return audio, out_path


def main():
    parser = argparse.ArgumentParser(description="Generate Gregory Gorin narration (Qwen3-TTS)")
    parser.add_argument(
        "--slide", type=int, choices=[1, 2, 3, 4],
        help="Generate a specific slide (default: all)",
    )
    parser.add_argument(
        "--play", action="store_true",
        help="Play audio after generation",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    slide_map = {1: "slide1_hook", 2: "slide2_audit", 3: "slide3_plan", 4: "slide4_cta"}
    if args.slide:
        to_generate = {slide_map[args.slide]: SLIDES[slide_map[args.slide]]}
    else:
        to_generate = SLIDES

    print(f"{'='*60}")
    print(f"  Gregory Gorin Narration — Qwen3-TTS Base ICL")
    print(f"  Slides: {', '.join(to_generate.keys())}")
    print(f"  Speed: {SPEED} | Mode: single-pass per slide")
    print(f"{'='*60}")

    print("\nLoading model...")
    t0 = time.time()
    model = load_model(MODEL_ID)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    all_audio = []
    all_paths = []

    for slide_name, text in to_generate.items():
        audio, path = generate_slide(model, slide_name, text)
        all_audio.append(audio)
        all_paths.append(path)

    # Concatenate full narration if multiple slides
    if len(all_audio) > 1:
        gap = np.zeros(int(SAMPLE_RATE * 0.8), dtype=np.float32)  # 800ms between slides
        combined = []
        for i, seg in enumerate(all_audio):
            combined.append(seg)
            if i < len(all_audio) - 1:
                combined.append(gap)
        full = np.concatenate(combined)
        full_path = OUTPUT_DIR / "qwen3_greg_full_narration.wav"
        sf.write(str(full_path), full, SAMPLE_RATE)
        print(f"\n{'='*60}")
        print(f"  Full narration: {full_path} ({len(full)/SAMPLE_RATE:.1f}s)")
        print(f"{'='*60}")

    if args.play:
        play_path = all_paths[0] if len(all_paths) == 1 else full_path
        print(f"\nPlaying: {play_path}")
        subprocess.run(["afplay", str(play_path)])

    print("\nDone!")


if __name__ == "__main__":
    main()
