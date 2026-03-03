# Voice Cloning Process — MOSS-TTS

Step-by-step process for cloning a real person's voice from a recording and generating natural-sounding narration.

## 1. Select a Reference Clip

Find a clean 10–15 second window from the source recording where the speaker is:
- Confident and natural (not hesitant or distracted)
- Clearly articulated with no filler words (um, uh)
- Not overlapping with other speakers
- Free of background noise, music, or system sounds
- Max gap between words ~0.5s (no long pauses)

Use the transcript (with word-level timestamps) to identify candidate windows and verify word-by-word.

## 2. Extract with ffmpeg

```bash
ffmpeg -i "<source_video_or_audio>" \
  -ss <start_seconds> -to <end_seconds> \
  -vn -ac 1 -ar 24000 -acodec pcm_s16le \
  "assets/audio/reference_<speaker>.wav"
```

Flags:
- `-vn` — strip video
- `-ac 1` — mono (required by MOSS-TTS)
- `-ar 24000` — 24kHz sample rate (matches model)
- `-acodec pcm_s16le` — 16-bit PCM WAV

Verify:
```bash
ffprobe -show_entries format=duration:stream=channels,sample_rate,codec_name \
  -of default=noprint_wrappers=1 assets/audio/reference_<speaker>.wav
```

Expected: ~10–15s, mono, 24000 Hz, pcm_s16le.

## 3. Write Conversational Script Text

**Critical:** The narration text determines naturalness more than any model parameter. Write it as speech, not prose.

### Do
- Use contractions: "you've got", "they're", "we're gonna"
- Add casual connectors: "So here's the thing", "Alright so", "And actually"
- Vary sentence length — mix short punchy lines with longer ones
- Use direct address: "All you gotta do is..."
- Include thinking-out-loud phrases: "which is interesting", "and the reason is"
- Match the speaker's actual vocabulary and cadence from the reference clip

### Don't
- Write formal/polished copy — it sounds robotic when spoken
- Use parallel structure for every sentence (sounds like a list)
- Stack long sentences back to back (monotone pacing)
- Use words the speaker wouldn't naturally say

### Example — Before (robotic)
> "Your reputation is strong. Your website isn't keeping up. Your Google Business Profile scores ninety out of one hundred."

### Example — After (natural)
> "So your reputation's great. Like, your Google Business Profile scores ninety out of a hundred, and you're already number one for dispensary Fishkill. That's solid. But here's what's costing you."

## 4. Add Emotion Instructions

MOSS-TTS `build_user_message()` accepts both `reference` (voice clone) and `instruction` (emotion/style direction) simultaneously.

```python
processor.build_user_message(
    text=script_text,
    reference=[ref_audio_path],
    instruction="Natural, conversational tone like you're walking a client through findings in a relaxed meeting."
)
```

### Guidelines for Instructions
- Keep them short and natural — one or two sentences
- Describe the *vibe*, not individual sentence beats
- Reference a relatable scenario ("like chatting with a friend", "like explaining results to a client you already know")
- Don't over-direct — the model gets confused with too many conflicting directives

### Per-slide examples (sales deck)
| Slide | Instruction |
|-------|-------------|
| Hook | "Natural, conversational tone like you're walking a client through findings in a relaxed meeting. Casual but confident." |
| Audit | "Friendly and direct, like explaining results to a client you already know. Matter-of-fact on the issues but not alarming." |
| Plan | "Upbeat and energetic like you're genuinely excited to share the game plan. Conversational momentum." |
| CTA | "Warm and easy-going, zero pressure. Like chatting with a friend about next steps." |

## 5. Generation Parameters

For voice cloning with MOSS-TTS, these params work well on Apple Silicon (MPS):

```python
model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=6144,
    audio_temperature=1.7,
    audio_top_p=0.8,
    audio_top_k=25,
    audio_repetition_penalty=1.0,
)
```

| Param | Value | Notes |
|-------|-------|-------|
| `audio_temperature` | 1.7 | Higher = more expressive. Below 1.5 tends monotone. |
| `audio_top_p` | 0.8 | Nucleus sampling threshold |
| `audio_top_k` | 25 | Limits token choices per step |
| `audio_repetition_penalty` | 1.0 | Keep at 1.0 for clone mode (penalty distorts voice match) |
| `max_new_tokens` | 6144 | Enough for ~30s of audio |

## 6. Run Generation

```bash
python narration_farmers_choice.py --voice adam_barrow          # sample (slide 1)
python narration_farmers_choice.py --voice adam_barrow --full    # all slides
python narration_farmers_choice.py --voice all --full            # all voices, all slides
```

Output lands in `narration_output/adam_barrow_slide1_hook.wav`, etc.

## 7. Iterate

If the output sounds off:

| Problem | Fix |
|---------|-----|
| Monotone / flat | Rewrite script to be more conversational; raise `audio_temperature` |
| Sentences run together | Add casual transitions between thoughts; shorter sentences |
| Sounds like reading a script | Use contractions, filler words, direct address |
| Voice doesn't match speaker | Use a longer/cleaner reference clip (12–15s ideal) |
| Too much variation / unstable | Lower `audio_temperature` toward 1.5 |
| Words garbled or repeated | Lower `audio_top_p` to 0.6; check for unusual words in script |

## File Reference

| File | Purpose |
|------|---------|
| `assets/audio/reference_adam_barrow.wav` | 14.8s reference clip from onboarding call |
| `narration_farmers_choice.py` | Generation script with all three voices |
| `narration_output/` | Generated audio files |
| `mps_compat.py` | MPS backend patches (auto-imported) |
| `optimized_generate.py` | Optimized generation loop |
