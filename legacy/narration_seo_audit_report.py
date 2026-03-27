"""
Generate SeoAuditReport narration in Adam Barrow's voice using MOSS-TTS.

Produces 10 scene audio files + manifest.json for Remotion integration.
Outputs directly to: ~/Projects/remotion/public/voiceover/SeoAuditReport/

Usage:
  python narration_seo_audit_report.py              # Sample mode (scene 0 only)
  python narration_seo_audit_report.py --full        # All 10 scenes
  python narration_seo_audit_report.py --full --skip-enhance  # Skip ffmpeg post-processing
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mps_compat  # noqa: F401, E402

from transformers import AutoModel, AutoProcessor

from optimized_generate import patch_generate

# ---------------------------------------------------------------------------
# Output directory — writes directly into Remotion's voiceover folder
# ---------------------------------------------------------------------------
REMOTION_VOICEOVER_DIR = (
    Path.home() / "Projects" / "remotion" / "public" / "voiceover" / "SeoAuditReport"
)

# ---------------------------------------------------------------------------
# 10 narration scripts rewritten in Adam Barrow's voice
# ---------------------------------------------------------------------------
NARRATION_SEGMENTS = [
    # Scene 0: Title
    (
        "scene-0",
        "Farmer's Choice Dispensary. S.E.O. Audit Report by Great Gateway.",
        "Professional and warm, like opening a presentation to a client you respect. "
        "Clear, measured, authoritative.",
    ),
    # Scene 1: SERP / Google Search
    (
        "scene-1",
        "So when someone searches 'dispensary fishkill n.y.' on Google, right? "
        "You're number one in the local map results, which is, you know, that's "
        "great. But in organic search, Fishkill Cannabis is outranking you. And "
        "with a website health score of just twenty eight out of a hundred, that "
        "number one spot, it's kind of at risk.",
        "Conversational and direct, like walking a client through their Google results "
        "on a shared screen. Natural pauses, thinking out loud.",
    ),
    # Scene 2: Heatmap
    (
        "scene-2",
        "So this heatmap shows how you rank for 'dispensary near me' across the "
        "Fishkill area, right? Green means you're number one, which is what we "
        "want everywhere. Orange and red, that's where your competitors are "
        "winning. You're visible in seventy four out of eighty one grid zones, "
        "but your average rank is only three point seven. And that forty three "
        "percent coverage gap, that means nearly half the Hudson Valley, you "
        "know, they can't find you.",
        "Analytical but accessible, like explaining a dashboard to a client. "
        "Slight concern on the gap numbers but not alarming.",
    ),
    # Scene 3: Gap / Paradox
    (
        "scene-3",
        "So here's, here's the paradox, right? You've got a four point seven "
        "star rating, three hundred and six reviews, nearly three thousand keyword "
        "rankings. But your website health score is just twenty eight out of a "
        "hundred. You know, the reputation is there, the website just isn't "
        "keeping up.",
        "Genuine surprise and empathy, like you're sharing a finding that doesn't "
        "make sense at first glance. Conversational, thinking out loud.",
    ),
    # Scene 4: Scores Breakdown
    (
        "scene-4",
        "Alright, so let's break that down. Overall: twenty eight. Performance: "
        "thirty. Technical S.E.O.: eighteen. Content: just fifteen. Local S.E.O. "
        "is your strongest at forty eight, but, you know, it's still below "
        "average. There's a lot of room to grow here.",
        "Matter-of-fact delivery, like reading scores to a client. Slight emphasis "
        "on the low numbers. Not judgmental, just honest.",
    ),
    # Scene 5: Strengths
    (
        "scene-5",
        "But the good news, right? Your foundation is solid. High ratings, "
        "strong reviews, thousands of keywords already ranking, and a good "
        "technical setup with H.T.T.P.S. and sitemaps. You know, we're not "
        "starting from scratch here. We're building on something real.",
        "Upbeat and encouraging, like delivering good news after tough findings. "
        "Genuine warmth, building confidence.",
    ),
    # Scene 6: Problems / Critical Issues
    (
        "scene-6",
        "But we found, like, six critical issues that are kind of dragging you "
        "down, right? Eleven broken menu links. Missing search descriptions on "
        "all seven pages. No structured data, which means Google can't really "
        "understand your content. Address mismatches across the web. A blank "
        "Google Business Profile description. And four pages missing their main "
        "headlines. You know, these are all fixable, but they're costing you "
        "customers right now.",
        "Direct and honest, listing issues like a mechanic showing you what needs "
        "fixing. Not alarming, but clear that action is needed.",
    ),
    # Scene 7: Competitors
    (
        "scene-7",
        "And your competitors aren't standing still, right? Fishkill Cannabis "
        "has basic S.E.O. in place and is already beating you in organic search. "
        "Root Nine in Wappingers Falls has a full S.E.O. program and they're "
        "targeting your Fishkill searches. And then Curaleaf, you know, they're "
        "a multi-state operator with enterprise level optimization. So the "
        "window to act is, it's kind of now.",
        "Slightly urgent but not pushy. Like alerting a friend that competitors "
        "are making moves. Natural concern.",
    ),
    # Scene 8: 90-Day Plan
    (
        "scene-8",
        "Alright, so here's your ninety day plan, right? Week one: stop the "
        "bleeding. We fix those broken links, add meta descriptions, set up "
        "structured data. Weeks two through four: we build the foundation. "
        "Your Google Business Profile gets optimized, page headlines go in, "
        "we set up performance monitoring. And then months two and three, "
        "that's where we really kind of dominate the market, right? We target "
        "searches beyond Fishkill, rank for product specific terms, and, you "
        "know, we put you on the map across the whole Hudson Valley.",
        "Energetic and confident, like revealing a game plan you're genuinely "
        "excited about. Building momentum through each phase.",
    ),
    # Scene 9: CTA
    (
        "scene-9",
        "So let's, let's put Farmer's Choice Dispensary on the map, right? "
        "All you gotta do is reply to greg at great gateway dot com, and "
        "we'll schedule a fifteen minute strategy call this week. Again, we "
        "only win if you win first.",
        "Warm and easy-going, zero pressure. Like wrapping up a friendly meeting. "
        "Genuine and relaxed with a smile in the voice.",
    ),
]


def get_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda:0"), torch.bfloat16, "flash_attention_2"
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.bfloat16, "sdpa"
    else:
        return torch.device("cpu"), torch.float32, "eager"


def decode_audio(messages, sample_rate):
    """Extract and normalize audio from model output."""
    audio = messages[0].audio_codes_list[0]
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().float().cpu().numpy()
    else:
        audio_np = np.asarray(audio, dtype=np.float32)
    if audio_np.ndim > 1:
        audio_np = audio_np.reshape(-1)
    return audio_np


def enhance_audio(input_path, output_path=None):
    """Light post-processing: remove rumble, slight warmth, soft compression, normalize."""
    if output_path is None:
        output_path = input_path
    tmp_path = str(input_path) + ".tmp.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path), "-af",
        ",".join([
            "highpass=f=60",
            "equalizer=f=200:t=q:w=1:g=1",
            "equalizer=f=3000:t=q:w=2:g=1.5",
            "acompressor=threshold=-24dB:ratio=2:attack=10:release=100:makeup=1",
            "alimiter=limit=0.95:attack=2:release=20",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
        ]),
        "-ar", "24000", "-ac", "1", "-acodec", "pcm_s16le", tmp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[enhance] Warning: ffmpeg failed: {result.stderr[:200]}")
        return
    Path(tmp_path).replace(output_path)
    print(f"[enhance] Enhanced: {output_path}")


def convert_to_mp3(wav_path, mp3_path):
    """Convert WAV to MP3 for Remotion consumption."""
    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-codec:a", "libmp3lame", "-b:a", "192k",
        str(mp3_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[convert] Warning: ffmpeg MP3 conversion failed: {result.stderr[:200]}")
        return False
    return True


def generate_adam_barrow(segments, output_dir, skip_enhance=False):
    """Generate narration using Adam Barrow voice clone via MOSS-TTS."""
    device, dtype, attn_impl = get_device_and_dtype()
    model_path = "OpenMOSS-Team/MOSS-TTS"
    ref_audio = str(
        Path(__file__).resolve().parent / "assets" / "audio" / "reference_adam_barrow.wav"
    )

    print(f"[Adam Barrow] Loading model on {device}...")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    ).to(device)
    model.eval()
    patch_generate(model)

    sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
    print(f"[Adam Barrow] Model loaded in {time.time() - t0:.1f}s")

    manifest = []

    for name, text, emotion in segments:
        print(f"\n[Adam Barrow] Generating: {name} ({len(text)} chars)")
        print(f"[Adam Barrow] Emotion: {emotion[:60]}...")

        conversations = [
            [
                processor.build_user_message(
                    text=text, reference=[ref_audio], instruction=emotion
                )
            ]
        ]
        batch = processor(conversations, mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=6144,
                audio_temperature=1.7,
                audio_top_p=0.8,
                audio_top_k=25,
                audio_repetition_penalty=1.0,
            )
        gen_time = time.time() - t0

        messages = processor.decode(outputs)
        audio_np = decode_audio(messages, sample_rate)
        duration = len(audio_np) / sample_rate
        print(f"[Adam Barrow] {name}: {gen_time:.1f}s generation -> {duration:.1f}s audio")

        # Save WAV
        wav_path = output_dir / f"{name}.wav"
        sf.write(str(wav_path), audio_np, sample_rate)

        # Enhance
        if not skip_enhance:
            enhance_audio(wav_path)

        # Convert to MP3
        mp3_path = output_dir / f"{name}.mp3"
        if convert_to_mp3(wav_path, mp3_path):
            print(f"[Adam Barrow] Saved: {mp3_path}")
            # Clean up WAV after successful MP3 conversion
            wav_path.unlink(missing_ok=True)
        else:
            print(f"[Adam Barrow] Kept WAV: {wav_path}")

        # Get actual duration from the audio
        duration_ms = int(duration * 1000)
        manifest.append({
            "index": int(name.split("-")[1]),
            "file": f"{name}.mp3",
            "durationMs": duration_ms,
            "durationSec": round(duration, 2),
            "text": text,
        })

    # Write manifest.json
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[Adam Barrow] Manifest written: {manifest_path}")

    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Generate SeoAuditReport narration in Adam Barrow's voice"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Generate all 10 scenes (default: scene 0 only)",
    )
    parser.add_argument(
        "--skip-enhance",
        action="store_true",
        help="Skip ffmpeg audio enhancement",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: Remotion voiceover dir)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else REMOTION_VOICEOVER_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = NARRATION_SEGMENTS if args.full else NARRATION_SEGMENTS[:1]

    mode_label = "FULL (all 10 scenes)" if args.full else "SAMPLE (scene 0 only)"
    print(f"\n{'=' * 60}")
    print("SeoAuditReport Narration — Adam Barrow Voice")
    print(f"Mode: {mode_label}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")

    manifest = generate_adam_barrow(
        segments, output_dir, skip_enhance=args.skip_enhance
    )

    print(f"\n{'=' * 60}")
    print(f"Done! {len(manifest)} scenes generated.")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
