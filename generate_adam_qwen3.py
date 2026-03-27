"""
Generate Adam Barrow voice clone narration using Qwen3-TTS Base ICL mode.

Chunk-based generation with breathing pauses, Adam-style tics, and
dynamic inflections for natural delivery.

Usage:
  cd /Users/gregorygorin/Projects/voice/voiceclone
  source .venv/bin/activate
  python generate_adam_qwen3.py                    # all slides
  python generate_adam_qwen3.py --slide 1          # slide 1 only
  python generate_adam_qwen3.py --slide 1 --play   # generate + play
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
REF_AUDIO = str(SCRIPT_DIR / "assets" / "audio" / "reference_adam_barrow.wav")
REF_TEXT = (
    "these are the type of email messages and like this will go again through "
    "an audit. It is allowed to you if you want us to audit and give a "
    "recommendation, we are happy to do it. If you are like, Adam, we are good, "
    "then you are good. And we will, we will just move forward as, as needed, "
    "but we are happy to do this free."
)
OUTPUT_DIR = SCRIPT_DIR / "narration_output"

# ─── Generation Settings ────────────────────────────────────────────────
MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
SPEED = 0.92
SAMPLE_RATE = 24000
PAUSE_MS = 450        # standard pause between chunks
SHORT_PAUSE_MS = 350  # shorter pause between list items

# ─── Slide Scripts (chunked for breathing pauses) ───────────────────────
# Each slide is a list of (chunk_text, pause_type) tuples.
# pause_type: "normal" = 450ms, "short" = 350ms, "none" = last chunk

SLIDES = {
    "slide1_hook": [
        ("So here's... here's what's really interesting, right?", "normal"),
        ("You've got three hundred and six reviews... at four point seven stars. That's HUGE.", "normal"),
        ("But here's the thing. That's one more review than Fishkill Cannabis... and they're still outranking you?", "normal"),
        ("And the reason is, you know, their website has what yours is missing.", "normal"),
        ("Meta descriptions.", "short"),
        ("Schema markup.", "short"),
        ("Kind of, you know, proper on-page S.E.O.", "normal"),
        ("And you've got the reviews. You've got the reputation. You've got the whole thing, right?", "normal"),
        ("Your website score's sitting at... forty two. Out of a hundred.", "normal"),
        ("But the good news? The fix is WAY faster than you'd think. And honestly, a lot of it's low-hanging fruit, right?", "none"),
    ],
    "slide2_audit": [
        ("So your reputation's great, right? Like, your Google Business Profile scores ninety out of a hundred.", "normal"),
        ("And you're already ranking number one for dispensary Fishkill. That's... that's solid.", "normal"),
        ("But here's what's kind of costing you customers.", "normal"),
        ("You've got zero meta descriptions. Across all fifteen pages.", "normal"),
        ("Your homepage doesn't have an H1 tag... so Google can't even tell what your main heading is, right?", "normal"),
        ("No schema markup. And your competitors? They have it.", "normal"),
        ("You know, there's like eleven broken links hurting your user experience.", "normal"),
        ("And then your homepage only has about two hundred fifty words... which Google sees as thin content.", "normal"),
        ("So again, there's... there's a lot of low-hanging fruit here.", "none"),
    ],
    "slide3_plan": [
        ("Alright, so here's the ninety day plan.", "normal"),
        ("We've got sixteen moves to get you that number one spot, right?", "normal"),
        ("And we're projecting your score goes from like forty two... up to between eighty and ninety. Which is HUGE.", "normal"),
        ("So phase one is stop the bleeding. This week.", "normal"),
        ("We add meta descriptions to all fifteen pages.", "short"),
        ("Get the H1 headings in.", "short"),
        ("Implement schema markup.", "short"),
        ("And fix those eleven broken links.", "normal"),
        ("Phase two, we, we build the foundation. We fix your N.A.P. inconsistencies, get you a Leafly profile, and build out a location landing page.", "normal"),
        ("And then phase three is where we really kind of dominate the market, right?", "normal"),
        ("We expand your content, create landing pages for Beacon, Wappingers, Newburgh... and launch a monthly blog.", "normal"),
        ("And, you know, the proof's in the pudding. We're, we're really excited about what this is gonna do for you guys.", "none"),
    ],
    "slide4_cta": [
        ("So let's, let's get Farmers Choice into that number one spot, right?", "normal"),
        ("All you gotta do is reply to this email. Say hey, I'm interested.", "normal"),
        ("No commitment. No pressure at all. We sit on the exact same side of the table here.", "normal"),
        ("Then we'll hop on a quick fifteen minute call and walk through the top five quick wins together.", "normal"),
        ("And from there, you know, we build... you grow. We handle all the technical stuff.", "normal"),
        ("You keep doing what you're great at, which is running Fishkill's best dispensary.", "normal"),
        ("And actually, before we even get on that call? We're gonna deliver three quick wins. Gratis.", "normal"),
        ("A custom Google Business Profile description.", "short"),
        ("Meta description copy for your top pages.", "short"),
        ("And a full list of every N.A.P. mismatch with the exact corrections.", "normal"),
        ("Again, we only win if you win first, right? So let's, let's get you guys turned on.", "none"),
    ],
}


def generate_slide(model, slide_name, chunks):
    """Generate a single slide with chunked pauses."""
    silence_normal = np.zeros(int(SAMPLE_RATE * PAUSE_MS / 1000), dtype=np.float32)
    silence_short = np.zeros(int(SAMPLE_RATE * SHORT_PAUSE_MS / 1000), dtype=np.float32)

    print(f"\n{'─'*60}")
    print(f"  {slide_name} ({len(chunks)} chunks)")
    print(f"{'─'*60}")

    audio_parts = []
    t_slide = time.time()

    for i, (text, pause_type) in enumerate(chunks):
        print(f"  [{i+1}/{len(chunks)}] {text[:65]}{'...' if len(text) > 65 else ''}")
        t1 = time.time()
        results = list(model.generate(
            text=text,
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
            speed=SPEED,
        ))
        chunk_audio = np.array(results[0].audio)
        audio_parts.append(chunk_audio)
        dur = len(chunk_audio) / SAMPLE_RATE
        print(f"         {time.time()-t1:.1f}s -> {dur:.1f}s audio")

        if pause_type == "normal":
            audio_parts.append(silence_normal)
        elif pause_type == "short":
            audio_parts.append(silence_short)
        # "none" = no pause (last chunk)

    final = np.concatenate(audio_parts)
    out_path = OUTPUT_DIR / f"qwen3_adam_{slide_name}.wav"
    sf.write(str(out_path), final, SAMPLE_RATE)

    total_dur = len(final) / SAMPLE_RATE
    elapsed = time.time() - t_slide
    print(f"\n  Saved: {out_path}")
    print(f"  {total_dur:.1f}s audio in {elapsed:.1f}s (RTF: {elapsed/total_dur:.1f}x)")

    return final, out_path


def main():
    parser = argparse.ArgumentParser(description="Generate Adam Barrow narration (Qwen3-TTS)")
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
    print(f"  Adam Barrow Narration — Qwen3-TTS Base ICL")
    print(f"  Slides: {', '.join(to_generate.keys())}")
    print(f"  Speed: {SPEED} | Pause: {PAUSE_MS}ms | List pause: {SHORT_PAUSE_MS}ms")
    print(f"{'='*60}")

    print("\nLoading model...")
    t0 = time.time()
    model = load_model(MODEL_ID)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    all_audio = []
    all_paths = []

    for slide_name, chunks in to_generate.items():
        audio, path = generate_slide(model, slide_name, chunks)
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
        full_path = OUTPUT_DIR / "qwen3_adam_full_narration.wav"
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
