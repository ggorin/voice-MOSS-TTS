"""
Generate narration samples for Farmers Choice sales deck.
Produces three versions:
  1. David Attenborough voice (Clone mode with reference audio)
  2. Custom consultant voice (MOSS-VoiceGenerator with instruction)
  3. Adam Barrow voice (Clone mode with onboarding call reference)

Usage:
  python narration_farmers_choice.py --voice attenborough
  python narration_farmers_choice.py --voice consultant
  python narration_farmers_choice.py --voice adam_barrow
  python narration_farmers_choice.py --voice both        # attenborough + consultant
  python narration_farmers_choice.py --voice all         # all three voices
"""

import argparse
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
# Narration script — one segment per slide
# ---------------------------------------------------------------------------
NARRATION_SEGMENTS = [
    # Slide 1: Hook
    (
        "slide1_hook",
        "So here's, here's what's really interesting, right? You've got three "
        "hundred and six reviews at four point seven stars. You know, that's one "
        "more review than Fishkill Cannabis. But they're still outranking you, "
        "and the reason is, you know, their website has what yours is missing. "
        "Meta descriptions, schema markup, kind of proper on-page S.E.O. "
        "And you've got the reviews, you've got the reputation, you've got the "
        "whole thing. Your website score's sitting at like forty two out of a "
        "hundred, right? But the good news is the fix is, it's way faster than "
        "you'd think.",
    ),
    # Slide 2: Audit findings
    (
        "slide2_audit",
        "So your reputation's great, right? Like, your Google Business Profile "
        "scores ninety out of a hundred, and you're already ranking number one "
        "for dispensary Fishkill. That's, that's solid. But here's what's kind "
        "of costing you customers. You've got zero meta descriptions across all "
        "fifteen pages. Your homepage doesn't have an H1 tag, so Google can't "
        "even tell what your main heading is, right? No schema markup, and your "
        "competitors, they have it. You know, there's like eleven broken links "
        "hurting your user experience. And then your homepage only has about two "
        "hundred fifty words, which Google sees as thin content. So again, "
        "there's, there's a lot of low-hanging fruit here.",
    ),
    # Slide 3: 90-day plan
    (
        "slide3_plan",
        "Alright, so here's the ninety day plan. We've got sixteen moves to get "
        "you that number one spot, right? And we're projecting your score goes "
        "from like forty two up to, you know, between eighty and ninety. So "
        "phase one is stop the bleeding. This week we add meta descriptions to "
        "all fifteen pages, get the H1 headings in, implement schema markup, "
        "and fix those eleven broken links. Phase two, we, we build the foundation. "
        "We fix your N.A.P. inconsistencies, get you a Leafly profile, and build "
        "out a location landing page. And then phase three is where we really "
        "kind of dominate the market, right? We expand your content, create "
        "landing pages for Beacon, Wappingers, Newburgh, and launch a monthly "
        "blog. And, you know, the proof's in the pudding. We're, we're really "
        "excited about what this is going to do for you guys.",
    ),
    # Slide 4: CTA
    (
        "slide4_cta",
        "So let's, let's get Farmers Choice into that number one spot, right? "
        "All you gotta do is reply to this email, say hey, I'm interested. No "
        "commitment, no pressure at all. We sit on the exact same side of the "
        "table here. Then we'll hop on a quick fifteen minute call and walk "
        "through the top five quick wins together. And from there, you know, "
        "we build, you grow. We handle all the technical stuff, you keep doing "
        "what you're great at, which is running Fishkill's best dispensary. "
        "And actually, before we even get on that call, we're gonna deliver "
        "three quick wins gratis. A custom Google Business Profile description, "
        "meta description copy for your top pages, and a full list of every "
        "N.A.P. mismatch with the exact corrections. Again, we only win if you "
        "win first, right? So let's, let's get you guys turned on.",
    ),
]

# For quick sample comparison, use just the hook
SAMPLE_TEXT = NARRATION_SEGMENTS[0][1]

OUTPUT_DIR = Path(__file__).resolve().parent / "narration_output"

# Voice instruction for the consultant voice (VoiceGenerator)
CONSULTANT_INSTRUCTION = (
    "A professional male consultant in his mid-thirties with a warm, confident American voice. "
    "Clear articulation with measured pacing, suitable for presenting business data and recommendations. "
    "Authoritative but approachable, like a trusted advisor walking you through important findings. "
    "Steady rhythm with slight emphasis on key numbers and action items."
)

# Per-slide emotion instructions for Adam Barrow voice clone
ADAM_BARROW_EMOTIONS = {
    "slide1_hook": (
        "Natural, conversational tone like you're walking a client through findings "
        "in a relaxed meeting. Casual but confident. Think out loud, not reading."
    ),
    "slide2_audit": (
        "Friendly and direct, like explaining results to a client you already know. "
        "Matter-of-fact on the issues but not alarming. Casual, natural pacing."
    ),
    "slide3_plan": (
        "Upbeat and energetic like you're genuinely excited to share the game plan. "
        "Conversational momentum, like you're talking someone through it live."
    ),
    "slide4_cta": (
        "Warm and easy-going, zero pressure. Like chatting with a friend about next steps. "
        "Genuine and relaxed, with a little excitement about the free deliverables."
    ),
}


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


def concatenate_segments(all_audio, sample_rate):
    """Join audio segments with 0.5s silence gaps."""
    silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    combined = []
    for i, seg in enumerate(all_audio):
        combined.append(seg)
        if i < len(all_audio) - 1:
            combined.append(silence)
    return np.concatenate(combined)


def enhance_audio(input_path, output_path=None):
    """Post-process audio to sound more professional.

    Applies: high-pass filter (remove rumble), presence EQ boost,
    gentle compression, limiting, and loudness normalization.
    """
    if output_path is None:
        output_path = input_path
    tmp_path = str(input_path) + ".tmp.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path), "-af",
        ",".join([
            "highpass=f=80",
            "lowpass=f=12000",
            "equalizer=f=180:t=q:w=1.5:g=1.5",
            "equalizer=f=3000:t=q:w=2:g=3",
            "equalizer=f=5000:t=q:w=2:g=2",
            "acompressor=threshold=-20dB:ratio=3:attack=5:release=50:makeup=1.5",
            "alimiter=limit=0.95:attack=1:release=10",
            "dynaudnorm=p=0.95:m=10",
        ]),
        "-ar", "24000", "-ac", "1", "-acodec", "pcm_s16le", tmp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[enhance] Warning: ffmpeg failed: {result.stderr[:200]}")
        return
    Path(tmp_path).replace(output_path)
    print(f"[enhance] Enhanced: {output_path}")


def generate_attenborough(segments, sample_only=True):
    """Generate narration using David Attenborough voice clone."""
    device, dtype, attn_impl = get_device_and_dtype()
    model_path = "OpenMOSS-Team/MOSS-TTS"
    ref_audio = str(
        Path(__file__).resolve().parent / "assets" / "audio" / "reference_en_2.mp3"
    )

    print(f"[Attenborough] Loading model on {device}...")
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
    print(f"[Attenborough] Model loaded in {time.time() - t0:.1f}s")

    to_generate = segments[:1] if sample_only else segments
    all_audio = []

    for name, text in to_generate:
        print(f"\n[Attenborough] Generating: {name} ({len(text)} chars)")
        conversations = [
            [processor.build_user_message(text=text, reference=[ref_audio])]
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
        elapsed = time.time() - t0

        messages = processor.decode(outputs)
        audio_np = decode_audio(messages, sample_rate)
        duration = len(audio_np) / sample_rate
        print(
            f"[Attenborough] {name}: {elapsed:.1f}s generation -> {duration:.1f}s audio"
        )

        out_path = OUTPUT_DIR / f"attenborough_{name}.wav"
        sf.write(str(out_path), audio_np, sample_rate)
        print(f"[Attenborough] Saved: {out_path}")
        all_audio.append(audio_np)

    if len(all_audio) > 1:
        full_audio = concatenate_segments(all_audio, sample_rate)
        full_path = OUTPUT_DIR / "attenborough_full_narration.wav"
        sf.write(str(full_path), full_audio, sample_rate)
        print(
            f"\n[Attenborough] Full narration saved: {full_path} "
            f"({len(full_audio) / sample_rate:.1f}s)"
        )

    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def generate_consultant(segments, sample_only=True):
    """Generate narration using MOSS-VoiceGenerator with instruction."""
    device, dtype, attn_impl = get_device_and_dtype()
    model_path = "OpenMOSS-Team/MOSS-VoiceGenerator"

    print(f"[Consultant] Loading VoiceGenerator model on {device}...")
    print("[Consultant] (First run will download the model)")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        normalize_inputs=True,
    )
    if hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModel.from_pretrained(model_path, **model_kwargs).to(device)
    model.eval()

    # Try to apply optimized generate patch
    try:
        patch_generate(model)
        print("[Consultant] Applied optimized generate patch")
    except Exception:
        print(
            "[Consultant] Note: optimized_generate patch not compatible, "
            "using default generate"
        )

    sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
    print(f"[Consultant] Model loaded in {time.time() - t0:.1f}s")

    to_generate = segments[:1] if sample_only else segments
    all_audio = []

    for name, text in to_generate:
        print(f"\n[Consultant] Generating: {name} ({len(text)} chars)")
        conversations = [
            [
                processor.build_user_message(
                    text=text, instruction=CONSULTANT_INSTRUCTION
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
                audio_temperature=1.5,
                audio_top_p=0.6,
                audio_top_k=50,
                audio_repetition_penalty=1.1,
            )
        elapsed = time.time() - t0

        messages = processor.decode(outputs)
        audio_np = decode_audio(messages, sample_rate)
        duration = len(audio_np) / sample_rate
        print(
            f"[Consultant] {name}: {elapsed:.1f}s generation -> {duration:.1f}s audio"
        )

        out_path = OUTPUT_DIR / f"consultant_{name}.wav"
        sf.write(str(out_path), audio_np, sample_rate)
        print(f"[Consultant] Saved: {out_path}")
        all_audio.append(audio_np)

    if len(all_audio) > 1:
        full_audio = concatenate_segments(all_audio, sample_rate)
        full_path = OUTPUT_DIR / "consultant_full_narration.wav"
        sf.write(str(full_path), full_audio, sample_rate)
        print(
            f"\n[Consultant] Full narration saved: {full_path} "
            f"({len(full_audio) / sample_rate:.1f}s)"
        )

    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def generate_adam_barrow(segments, sample_only=True):
    """Generate narration using Adam Barrow voice clone."""
    device, dtype, attn_impl = get_device_and_dtype()
    model_path = "OpenMOSS-Team/MOSS-TTS"
    ref_audio = str(
        Path(__file__).resolve().parent
        / "assets"
        / "audio"
        / "reference_adam_barrow_enhanced.wav"
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

    to_generate = segments[:1] if sample_only else segments
    all_audio = []

    for name, text in to_generate:
        emotion = ADAM_BARROW_EMOTIONS.get(name)
        print(f"\n[Adam Barrow] Generating: {name} ({len(text)} chars)")
        if emotion:
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
        elapsed = time.time() - t0

        messages = processor.decode(outputs)
        audio_np = decode_audio(messages, sample_rate)
        duration = len(audio_np) / sample_rate
        print(
            f"[Adam Barrow] {name}: {elapsed:.1f}s generation -> {duration:.1f}s audio"
        )

        out_path = OUTPUT_DIR / f"adam_barrow_{name}.wav"
        sf.write(str(out_path), audio_np, sample_rate)
        enhance_audio(out_path)
        print(f"[Adam Barrow] Saved: {out_path}")
        all_audio.append(audio_np)

    if len(all_audio) > 1:
        full_audio = concatenate_segments(all_audio, sample_rate)
        full_path = OUTPUT_DIR / "adam_barrow_full_narration.wav"
        sf.write(str(full_path), full_audio, sample_rate)
        enhance_audio(full_path)
        print(
            f"\n[Adam Barrow] Full narration saved: {full_path} "
            f"({len(full_audio) / sample_rate:.1f}s)"
        )

    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Generate Farmers Choice narration samples"
    )
    parser.add_argument(
        "--voice",
        choices=["attenborough", "consultant", "adam_barrow", "both", "all"],
        default="both",
        help="Which voice to generate (both=attenborough+consultant, all=all three)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Generate all slides (default: sample only - slide 1 hook)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_only = not args.full

    mode_label = "SAMPLE (slide 1 only)" if sample_only else "FULL (all slides)"
    print(f"\n{'=' * 60}")
    print("Farmers Choice Sales Deck Narration")
    print(f"Mode: {mode_label}")
    print(f"{'=' * 60}\n")

    if args.voice in ("attenborough", "both", "all"):
        generate_attenborough(NARRATION_SEGMENTS, sample_only=sample_only)

    if args.voice in ("consultant", "both", "all"):
        generate_consultant(NARRATION_SEGMENTS, sample_only=sample_only)

    if args.voice in ("adam_barrow", "all"):
        generate_adam_barrow(NARRATION_SEGMENTS, sample_only=sample_only)

    print(f"\n{'=' * 60}")
    print(f"Done! Output files in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
