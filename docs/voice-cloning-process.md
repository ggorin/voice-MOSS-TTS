# Voice Cloning Process

Complete guide for cloning voices and generating natural-sounding narration using Qwen3-TTS Base ICL mode. Covers both the original Adam Barrow workflow and the 22-voice celebrity framework.

> **Recommended model: Qwen3-TTS Base (ICL mode)** — cleaner output (34 dB SNR vs 18 dB for MOSS-TTS), better voice fidelity, natural pacing. MOSS-TTS workflow kept at the bottom as a legacy reference.

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Reference Audio Setup](#1-reference-audio-setup)
3. [Write Conversational Script Text](#2-write-conversational-script-text)
4. [Embed Speaking Tics](#3-embed-speaking-tics)
5. [Control Voice Inflection Through Text](#4-control-voice-inflection-through-text)
6. [Chunk-Based Generation with Breathing Pauses](#5-chunk-based-generation-with-breathing-pauses)
7. [Pause Types: Normal vs Short](#6-pause-types-normal-vs-short)
8. [Full Worked Example: Slide 1 Hook](#7-full-worked-example-slide-1-hook)
9. [Generation Script](#8-generation-script)
10. [Generation Parameters](#9-generation-parameters)
11. [Model Comparison](#10-model-comparison)
12. [Environment Setup](#11-environment-setup)
13. [Troubleshooting](#12-troubleshooting)

---

## Quick Start

```bash
# From the mlx-tts project directory
cd /Users/gregorygorin/Projects/voice/mlx-tts
source .venv/bin/activate

# Generate all 4 slides
python ../MOSS-TTS/generate_adam_qwen3.py

# Generate slide 1 only and play it
python ../MOSS-TTS/generate_adam_qwen3.py --slide 1 --play
```

---

## 1. Reference Audio Setup

### Source
- **Recording:** `dank_you_onboarding.mp4` (Adam Barrow's onboarding call, ~20 min)
- **Reference clip:** `assets/audio/reference_adam_barrow.wav`
  - 14.8s, mono, 24kHz, PCM 16-bit
  - Timestamp: 379.0s–393.8s from the onboarding video
  - Content: Adam explaining the audit/recommendation process — confident, professional, natural cadence

### Reference clip transcript (required by Qwen3 ICL mode)
```
these are the type of email messages and like this will go again through an audit.
It's allowed to you if you want us to audit and give a recommendation, we're happy
to do it. If you're like, Adam, we're good, then you're good. And we'll, we'll just
move forward as, as needed, but we're happy to do this free.
```

> **Important:** Qwen3 ICL mode requires both `ref_audio` AND `ref_text`. The transcript must match the audio closely — the model uses it to align the voice characteristics with the text tokens.

### Selecting a good reference clip
Find a clean 10–15 second window where the speaker is:
- Confident and natural (not hesitant or distracted)
- Clearly articulated with no filler words (um, uh)
- Not overlapping with other speakers
- Free of background noise, music, or system sounds
- Max gap between words ~0.5s (no long pauses)

Use the transcript (with word-level timestamps) to identify candidate windows and verify word-by-word.

### Extract with ffmpeg
```bash
ffmpeg -i "/path/to/source_video.mp4" \
  -ss 379.0 -to 393.8 \
  -vn -ac 1 -ar 24000 -acodec pcm_s16le \
  "assets/audio/reference_adam_barrow.wav"
```

| Flag | Purpose |
|------|---------|
| `-vn` | Strip video track |
| `-ac 1` | Convert to mono (required by model) |
| `-ar 24000` | Resample to 24kHz (matches model's sample rate) |
| `-acodec pcm_s16le` | 16-bit PCM WAV format |

---

## 2. Write Conversational Script Text

The narration text determines naturalness more than any model parameter. Write it as speech, not prose.

See [adam-barrow-voice-profile.md](adam-barrow-voice-profile.md) for Adam's full speaking style reference (tics, signature phrases, pacing patterns, contractions, number handling).

### Rules

| Do | Don't |
|----|-------|
| Use contractions: "you've got", "they're", "we're gonna" | Write formal/polished copy — sounds robotic |
| Add casual connectors: "So here's the thing", "Alright so" | Use parallel structure for every sentence (sounds like a list) |
| Vary sentence length — mix short punchy lines with longer ones | Stack long sentences back to back (monotone pacing) |
| Use direct address: "All you gotta do is..." | Use words the speaker wouldn't naturally say |
| Include thinking-out-loud phrases: "and the reason is" | Use precise numbers — Adam rounds everything |

### Example transformation

**Original marketing copy (robotic when spoken):**
> "Your website scores forty-two out of one hundred. Your competitors have schema markup and meta descriptions that you're missing. We can fix this within ninety days."

**Rewritten for Adam's voice (natural when spoken):**
> "So your website's sitting at like forty-two out of a hundred, right? And your competitors, they've got schema markup, they've got meta descriptions, stuff that you're missing. But the good news is, we can get this fixed, you know, within ninety days. And honestly, a lot of it's low-hanging fruit."

**What changed:**
- Added "So" opener (Adam always opens with connectors)
- "like forty-two" instead of precise "forty-two" (Adam hedges numbers)
- Added "right?" as rhetorical check-in
- "they've got... they've got..." repetition (Adam's natural pattern)
- "you know" filler between clauses
- "the good news is" pivot (Adam always frames positively)
- "honestly" + "low-hanging fruit" (Adam's vocabulary)

---

## 3. Embed Speaking Tics

Qwen3 ICL clones the voice **timbre** (pitch, tone, resonance) but smooths out natural speech patterns. Without explicit tics in the text, the output sounds like a stranger reading Adam's words in Adam's voice. To get Adam's **authentic delivery**, embed his tics directly in the text.

### Tic Reference (ranked by frequency)

#### 1. `right?` — His #1 tic
Appended to the end of statements as a rhetorical confirmation. Use every 3–4 sentences.

```
❌ "They're still outranking you."
✅ "They're still outranking you, right?"

❌ "Your website score is sitting at forty-two out of a hundred."
✅ "Your website score's sitting at forty-two out of a hundred, right?"
```

#### 2. `you know` — Bridge filler
Connects clauses within a sentence. Appears multiple times per paragraph.

```
❌ "Their website has what yours is missing."
✅ "You know, their website has what yours is missing."

❌ "We can get this fixed within ninety days."
✅ "We can get this fixed, you know, within ninety days."
```

#### 3. `kind of` — Softener
Used before technical or assertive claims to keep things casual.

```
❌ "Proper on-page SEO."
✅ "Kind of, you know, proper on-page S.E.O."

❌ "We dominate the market."
✅ "We really kind of dominate the market."
```

#### 4. `...` (ellipsis) — Thinking pause
Signals the model to insert a micro-hesitation. Critical for making speech sound unrehearsed.

```
❌ "You've got three hundred and six reviews at four point seven stars."
✅ "You've got three hundred and six reviews... at four point seven stars."

❌ "So here's what's really interesting."
✅ "So here's... here's what's really interesting."
```

#### 5. Self-correction — Repeated phrase starts
Adam frequently restarts phrases when collecting thoughts. This builds emphasis and sounds natural.

```
❌ "Let's get Farmers Choice into that number one spot."
✅ "So let's, let's get Farmers Choice into that number one spot."

❌ "There's a lot of low-hanging fruit here."
✅ "There's... there's a lot of low-hanging fruit here."

❌ "We build the foundation."
✅ "We, we build the foundation."
```

#### 6. `again` — Re-emphasis device
Sprinkled in when reinforcing a point that was made earlier.

```
❌ "We only win if you win first."
✅ "Again, we only win if you win first."
```

### Before/After: Full paragraph

**Without tics (sounds like a stranger reading Adam's words):**
> "You've got three hundred and six reviews at four point seven stars. That's one more review than Fishkill Cannabis. But they're still outranking you because their website has what yours is missing. Meta descriptions, schema markup, proper on-page SEO."

**With tics (sounds like Adam):**
> "You've got, you know, three hundred and six reviews... at four point seven stars. And that's... that's one more review than Fishkill Cannabis. But they're still outranking you, right? And, and the reason is... you know, their website has what yours is missing. Meta descriptions. Schema markup. Kind of, you know, proper on-page S.E.O."

---

## 4. Control Voice Inflection Through Text

Qwen3 ICL has no `instruction` or `emotion` parameter. Instead, you control inflection entirely through how you write the text. The model interprets punctuation, capitalization, sentence structure, and word choice as delivery cues.

### Excitement / Energy

Use ALL CAPS for emphasis on key words. The model will stress these words.

```
❌ "The fix is faster than you'd think."
✅ "The fix is WAY faster than you'd think."

❌ "That's a big deal."
✅ "That's HUGE."

❌ "Which is exciting."
✅ "Which is HUGE."
```

Use exclamation marks sparingly for genuine energy:
```
✅ "That's one more review than Fishkill Cannabis!"
```

Use Adam's excitement words: "stoked", "love that", "really exciting", "off to the races"

### Emphasis / Dramatic Pause

Split a single thought into two short fragments separated by a period. The chunk boundary (or period within a chunk) creates a dramatic beat.

```
❌ "Your website score's sitting at forty-two out of a hundred."
✅ "Your website score's sitting at... forty two. Out of a hundred."
     ─────────────────────────────── ^^^^^^^^  ^^^^^^^^^^^^^^^^^^
     ellipsis = thinking pause        beat 1      beat 2 (lands hard)
```

Another example — splitting for dramatic reveal:
```
❌ "So phase one is stop the bleeding this week."
✅ "So phase one is stop the bleeding. This week."
                                       ^^^^^^^^^^
                                       lands as its own beat
```

### Curiosity / Disbelief

End statements with a question mark to create a rising intonation, as if Adam can't believe it himself:

```
❌ "They're still outranking you."
✅ "They're still outranking you?"
                                 ^
                                 rising tone = "can you believe this?"
```

```
❌ "And before we even get on that call, we're gonna deliver three quick wins."
✅ "And actually, before we even get on that call? We're gonna deliver three quick wins."
                                                ^
                                                pause + rising tone builds anticipation
```

### Warmth / Reassurance

Use longer, rolling sentences with softeners and "and" chains. This is Adam's natural reassurance mode:

```
✅ "No commitment. No pressure at all. We sit on the exact same side of the table here."
```

Use Adam's reassurance phrases: "we want you to feel comfortable", "no pressure at all", "we're here"

### Urgency / Gravity

Use short, staccato fragments. Each one lands like a separate point:

```
❌ "You've got zero meta descriptions, no H1 tag, and no schema markup."
✅ "Zero meta descriptions. No H1 tag. No schema markup."
    ^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^  ^^^^^^^^^^^^^^^^^
    boom                    boom        boom
```

List items as separate chunks (see next section) amplifies this even further.

### Casualness / Thinking Out Loud

Ellipses + "you know" + self-corrections signal unrehearsed thought:

```
✅ "And the reason is, you know, their website has what yours is missing."
✅ "But... but the good news is, the fix is... it's way faster than you'd think."
```

### Summary: Inflection Toolkit

| Inflection | Text Technique | Example |
|-----------|---------------|---------|
| **Excitement** | ALL CAPS on key word | "That's HUGE." |
| **Dramatic pause** | Split into two fragments | "Forty two. Out of a hundred." |
| **Curiosity/disbelief** | Question mark on statement | "they're still outranking you?" |
| **Warmth** | Long rolling sentence + softeners | "No commitment. No pressure at all." |
| **Urgency** | Short staccato fragments | "Zero meta descriptions. No H1 tag." |
| **Thinking out loud** | Ellipsis + self-correction | "But... but the good news is..." |
| **Emphasis** | Repetition (anaphora) | "You've got the reviews. You've got the reputation." |
| **Building anticipation** | Question mark mid-sentence + reveal | "before we get on that call? Three quick wins." |

---

## 5. Chunk-Based Generation with Breathing Pauses

This is the core technique that makes the narration sound natural instead of rushed and robotic.

### The Problem

When you generate a long paragraph in a single call, the model reads straight through with no breathing pauses. It sounds like someone reading a teleprompter at speed — unnatural, tiring to listen to, and nothing like how Adam actually speaks.

**Single-shot generation (bad):**
```python
# ❌ Everything in one call — runs together, no pauses
results = model.generate(
    text="So here's what's really interesting, right? You've got three hundred "
         "and six reviews at four point seven stars. That's one more review than "
         "Fishkill Cannabis. But they're still outranking you...",
    ref_audio=REF_AUDIO, ref_text=REF_TEXT, speed=0.92,
)
```

### The Solution

Break the text into **breath groups** — natural pause points where a real speaker would inhale. Generate each chunk separately, then stitch them together with silence gaps.

**Chunk-based generation (good):**
```python
# ✅ Each chunk = one breath group, stitched with silence
chunks = [
    "So here's... here's what's really interesting, right?",
    "You've got three hundred and six reviews... at four point seven stars. That's HUGE.",
    "But here's the thing. That's one more review than Fishkill Cannabis... "
    "and they're still outranking you?",
]

silence = np.zeros(int(24000 * 0.45), dtype=np.float32)  # 450ms

audio_parts = []
for chunk in chunks:
    results = list(model.generate(
        text=chunk, ref_audio=REF_AUDIO, ref_text=REF_TEXT, speed=0.92,
    ))
    audio_parts.append(np.array(results[0].audio))
    audio_parts.append(silence)

final = np.concatenate(audio_parts[:-1])  # trim trailing silence
```

### How to Identify Breath Group Boundaries

A breath group is 1–2 sentences that a speaker would say in one exhale before pausing to inhale. Break at:

1. **After "right?"** — Adam's natural check-in pause
2. **After a period that ends a thought** — complete idea = breath
3. **Before a topic shift** — "But here's the thing." starts a new thought
4. **Before/after a list** — the intro line is its own chunk, then each list item
5. **After an emotional beat** — "That's HUGE." deserves a moment to land

### Example: Identifying breath groups in raw text

**Raw text (one paragraph):**
> So here's what's really interesting, right? You've got three hundred and six reviews at four point seven stars. That's HUGE. But here's the thing. That's one more review than Fishkill Cannabis and they're still outranking you? And the reason is, you know, their website has what yours is missing. Meta descriptions. Schema markup. Kind of, you know, proper on-page S.E.O.

**Chunked into breath groups:**

| # | Chunk | Why break here |
|---|-------|----------------|
| 1 | "So here's... here's what's really interesting, right?" | After "right?" — natural check-in |
| 2 | "You've got three hundred and six reviews... at four point seven stars. That's HUGE." | Complete stat + reaction — let it land |
| 3 | "But here's the thing. That's one more review than Fishkill Cannabis... and they're still outranking you?" | Topic shift ("But") + question = pause |
| 4 | "And the reason is, you know, their website has what yours is missing." | Explanation intro — sets up the list |
| 5 | "Meta descriptions." | List item 1 (own chunk for emphasis) |
| 6 | "Schema markup." | List item 2 |
| 7 | "Kind of, you know, proper on-page S.E.O." | List item 3 (with softener) |

---

## 6. Pause Types: Normal vs Short

Not all pauses are equal. We use two pause lengths to create natural rhythm.

### Normal pause (450ms)
A full breath between distinct thoughts. Used between most chunks.

```python
("So here's... here's what's really interesting, right?", "normal"),
#                                                          ^^^^^^
#                                                          450ms gap → next thought
("You've got three hundred and six reviews...", "normal"),
```

### Short pause (350ms)
A quick beat between items in a list. Shorter because the speaker is enumerating, not switching topics.

```python
("Meta descriptions.", "short"),        # 350ms → quick beat
("Schema markup.", "short"),            # 350ms → quick beat
("Kind of, you know, proper on-page S.E.O.", "normal"),  # 450ms → thought complete
```

### No pause
Last chunk in a slide — no trailing silence needed.

```python
("But the good news? The fix is WAY faster than you'd think.", "none"),
#                                                               ^^^^
#                                                               end of slide
```

### When to use which

| Pause Type | Duration | Use When |
|-----------|----------|----------|
| `"normal"` | 450ms | Between distinct thoughts, after "right?", after emotional beats |
| `"short"` | 350ms | Between list items, rapid-fire points, items in a sequence |
| `"none"` | 0ms | Last chunk in a slide |

### Slide-level gap

When concatenating multiple slides into a full narration, use an 800ms gap between slides — longer than a breath pause to signal a section transition.

```python
slide_gap = np.zeros(int(24000 * 0.8), dtype=np.float32)  # 800ms between slides
```

---

## 7. Full Worked Example: Slide 1 Hook

This is the complete transformation from marketing copy to finished Adam Barrow narration.

### Step 1: Start with the raw data points

```
- 306 reviews, 4.7 stars
- 1 more review than Fishkill Cannabis competitor
- Competitor outranks them because of better on-page SEO
- Missing: meta descriptions, schema markup, on-page SEO
- Website score: 42/100
- Fix is straightforward
```

### Step 2: Write in Adam's voice (see [voice profile](adam-barrow-voice-profile.md))

```
So here's what's really interesting, right? You've got three hundred
and six reviews at four point seven stars. That's huge. But that's one
more review than Fishkill Cannabis, and they're still outranking you.
The reason is their website has what yours is missing. Meta descriptions,
schema markup, proper on-page SEO. You've got the reviews, you've got
the reputation, you've got the whole thing. Your website score's sitting
at forty two out of a hundred. But the good news is the fix is way
faster than you'd think.
```

### Step 3: Add speaking tics

```diff
- So here's what's really interesting, right?
+ So here's... here's what's really interesting, right?
                 ^^^^^^^^^ self-correction

- You've got three hundred and six reviews at four point seven stars.
+ You've got three hundred and six reviews... at four point seven stars.
                                           ^^^ thinking pause

- That's one more review than Fishkill Cannabis, and they're still outranking you.
+ That's one more review than Fishkill Cannabis... and they're still outranking you?
                                               ^^^ pause                          ^ disbelief

- The reason is their website has what yours is missing.
+ And the reason is, you know, their website has what yours is missing.
                     ^^^^^^^^^ filler

- Meta descriptions, schema markup, proper on-page SEO.
+ Meta descriptions. Schema markup. Kind of, you know, proper on-page S.E.O.
                   ^              ^  ^^^^^^^ softener     ^^^^^^^^^ spelled out
                   periods = separate items

- You've got the reviews, you've got the reputation
+ And you've got the reviews. You've got the reputation. You've got the whole thing, right?
  ^^^ connector             ^                          ^ periods for emphasis        ^^^^^^ tic

- Your website score's sitting at forty two out of a hundred.
+ Your website score's sitting at... forty two. Out of a hundred.
                              ^^^ dramatic     ^ split for emphasis

- But the good news is the fix is way faster than you'd think.
+ But the good news? The fix is WAY faster than you'd think.
                   ^ anticipation   ^^^ caps emphasis
  And honestly, a lot of it's low-hanging fruit, right?
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ added reassurance line
```

### Step 4: Add inflection markers

```
"That's HUGE."              ← caps = excitement on the stat
"and they're still          ← question mark = rising tone,
  outranking you?"             disbelief
"forty two. Out of a        ← split fragments = dramatic
  hundred."                    emphasis
"The fix is WAY faster"     ← caps = energy on the good news
```

### Step 5: Break into chunks with pause types

```python
SLIDE1_HOOK = [
    # Opener — pull them in with curiosity
    ("So here's... here's what's really interesting, right?", "normal"),

    # Key stat — let the number land, then react
    ("You've got three hundred and six reviews... at four point seven stars. That's HUGE.", "normal"),

    # Contrast — build tension with disbelief
    ("But here's the thing. That's one more review than Fishkill Cannabis... "
     "and they're still outranking you?", "normal"),

    # Explanation intro — sets up the list
    ("And the reason is, you know, their website has what yours is missing.", "normal"),

    # List items — each gets its own chunk + short pause
    ("Meta descriptions.", "short"),        # 350ms beat
    ("Schema markup.", "short"),            # 350ms beat
    ("Kind of, you know, proper on-page S.E.O.", "normal"),  # 450ms — list complete

    # Reassurance — validate what they DO have
    ("And you've got the reviews. You've got the reputation. "
     "You've got the whole thing, right?", "normal"),

    # The problem stat — dramatic split
    ("Your website score's sitting at... forty two. Out of a hundred.", "normal"),

    # Punchline — energy + warmth
    ("But the good news? The fix is WAY faster than you'd think. "
     "And honestly, a lot of it's low-hanging fruit, right?", "none"),
]
```

### Step 6: Generate

```bash
cd /Users/gregorygorin/Projects/voice/mlx-tts
source .venv/bin/activate
python ../MOSS-TTS/generate_adam_qwen3.py --slide 1 --play
```

### Result

| Metric | Value |
|--------|-------|
| Total chunks | 10 |
| Audio duration | ~33s |
| Generation time | ~60s on M4 Pro |
| SNR | ~34 dB |
| Output | `narration_output/qwen3_adam_slide1_hook.wav` |

---

## 8. Generation Script

The full generation script lives at `generate_adam_qwen3.py` in the project root.

### Usage

```bash
cd /Users/gregorygorin/Projects/voice/mlx-tts
source .venv/bin/activate

# All 4 slides + concatenated full narration
python ../MOSS-TTS/generate_adam_qwen3.py

# Single slide
python ../MOSS-TTS/generate_adam_qwen3.py --slide 1

# Generate and immediately play
python ../MOSS-TTS/generate_adam_qwen3.py --slide 1 --play
```

### Output files

| File | Content |
|------|---------|
| `narration_output/qwen3_adam_slide1_hook.wav` | Slide 1: Hook (competitive gap) |
| `narration_output/qwen3_adam_slide2_audit.wav` | Slide 2: Audit findings |
| `narration_output/qwen3_adam_slide3_plan.wav` | Slide 3: 90-day plan |
| `narration_output/qwen3_adam_slide4_cta.wav` | Slide 4: Call to action |
| `narration_output/qwen3_adam_full_narration.wav` | All 4 slides concatenated (800ms gaps) |

### Script structure

```
generate_adam_qwen3.py
├── REF_AUDIO, REF_TEXT       # Reference clip + transcript
├── SPEED, PAUSE_MS, SHORT_PAUSE_MS  # Generation settings
├── SLIDES dict               # All 4 slides as (chunk, pause_type) tuples
│   ├── slide1_hook (10 chunks)
│   ├── slide2_audit (9 chunks)
│   ├── slide3_plan (12 chunks)
│   └── slide4_cta (11 chunks)
├── generate_slide()          # Generate one slide from chunks
└── main()                    # CLI entry point
```

### Adding a new slide

1. Write the raw text using Adam's voice patterns
2. Add tics (right?, you know, ellipses, self-corrections)
3. Add inflection markers (CAPS, question marks, fragment splits)
4. Break into breath-group chunks with pause types
5. Add to the `SLIDES` dict in `generate_adam_qwen3.py`

```python
SLIDES["slide5_followup"] = [
    ("So here's, here's where we're at after week one, right?", "normal"),
    ("We got all fifteen meta descriptions live.", "short"),
    ("H1 tags are in.", "short"),
    ("Schema markup is done.", "normal"),
    ("And your score already jumped from forty two... to sixty eight.", "normal"),
    ("That's a twenty six point jump. In one week.", "normal"),
    ("And we're, we're just getting started, right?", "none"),
]
```

---

## 9. Generation Parameters

### Qwen3-TTS Base ICL (recommended)

| Param | Value | Notes |
|-------|-------|-------|
| `ref_audio` | Path to 10–15s reference WAV | Must be clean, mono, 24kHz |
| `ref_text` | Transcript of reference clip | Required for ICL mode — must closely match audio |
| `speed` | 0.92 | Slightly slower = more natural. Default 1.0 sounds rushed. |
| Model ID | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | ~3.4GB download on first run |
| Pause (normal) | 450ms | Between distinct thoughts |
| Pause (short) | 350ms | Between list items |
| Pause (slide gap) | 800ms | Between slides in full narration |

### Speed guide

| Speed | Effect |
|-------|--------|
| 1.0 | Default — sounds slightly rushed for narration |
| 0.92 | Sweet spot — natural, conversational pacing |
| 0.85 | Noticeably slow — use for emphasis-heavy sections |
| 0.80 | Very deliberate — sounds like a dramatic reading |

---

## 10. Model Comparison

We tested multiple TTS models for Adam's voice clone. Key findings:

| Model | SNR (dB) | Voice Match | Naturalness | Notes |
|-------|----------|-------------|-------------|-------|
| **Qwen3-TTS Base ICL** | **34.1** | Good | Good with tics | **Recommended.** Cleanest output. |
| MOSS-TTS | 18.8 | Very good | Good | Best timbre match but noisy. |
| Qwen3-TTS presets | 32–60 | N/A (not clone) | Excellent | No cloning, preset voices only. |
| Kokoro (am_adam) | 55.6 | N/A (not clone) | Truncates long text | Not suitable for narration. |

### Why Qwen3 over MOSS-TTS?
- **15+ dB better SNR** — MOSS-TTS has an inherent noise floor from its codec that can't be fixed with post-processing
- Every DSP processing stage (EQ, exciter, compression) we tried on MOSS-TTS output amplified the noise
- Minimal processing on MOSS-TTS helped marginally but couldn't overcome the fundamental SNR gap
- Qwen3 ICL clones timbre well enough while being dramatically cleaner
- No post-processing needed with Qwen3 — raw output is broadcast-ready

---

## 11. Environment Setup

### Qwen3-TTS (recommended)
```bash
cd /Users/gregorygorin/Projects/voice/mlx-tts
source .venv/bin/activate
# Model: mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16
# Auto-downloads ~3.4GB on first run
```

### MOSS-TTS (legacy)
```bash
cd /Users/gregorygorin/Projects/voice/MOSS-TTS
source .venv/bin/activate
python narration_farmers_choice.py --voice adam_barrow
```

---

## 12. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Monotone / flat delivery | Text is too formal, no variation | Rewrite with tics, ellipses, self-corrections, mixed sentence lengths |
| Sentences run together, no breathing | Generated as single long text | Use chunk-based generation with 450ms silence gaps |
| Reads list too fast | List items in same chunk | Split each list item into its own chunk with `"short"` pause type |
| Sounds like reading a script | Missing natural speech markers | Embed "right?", "you know", ellipses, self-corrections in text |
| Voice doesn't match Adam | Poor reference clip | Use a longer/cleaner reference clip (12–15s ideal) |
| Too fast / rushed | Default speed too high | Lower speed to 0.90–0.92 |
| No excitement on key points | No emphasis markers | Use ALL CAPS on key words: "That's HUGE", "WAY faster" |
| Everything same energy level | No inflection variation | Mix techniques: caps for excitement, fragments for drama, questions for curiosity |
| Noisy output (MOSS-TTS) | Codec noise floor | Switch to Qwen3-TTS Base ICL — inherently 15+ dB better SNR |
| Audio sounds like cheap mic | SNR problem, not bandwidth | Don't add EQ/exciter — switch models entirely |

---

## File Reference

| File | Purpose |
|------|---------|
| `generate_adam_qwen3.py` | Adam Barrow generation script (Qwen3-TTS, chunked, all 4 slides) |
| `generate_greg_qwen3.py` | Gregory Gorin generation script (Qwen3-TTS, single-pass) |
| `generate_celebrity.py` | Unified celebrity voice generation CLI (22 voices) |
| `tools/extract_reference.py` | YouTube reference audio extraction helper |
| `celebrities/configs/*.yaml` | Per-celebrity YAML configs (speed, mode, ref audio) |
| `celebrities/profiles/*.md` | Voice profile guides (tics, patterns, pacing) |
| `celebrities/scripts/*.yaml` | Monologue scripts per celebrity |
| `celebrities/audio/*.wav` | Reference audio clips (24kHz mono) |
| `assets/audio/reference_adam_barrow.wav` | 14.8s reference clip from onboarding call |
| `docs/adam-barrow-voice-profile.md` | Adam's full speaking style & tics guide |
| `docs/voice-cloning-process.md` | This document |
| `narration_farmers_choice.py` | MOSS-TTS generation script (legacy) |
| `narration_output/` | All generated audio files |
| `studio_post_process.py` | DSP post-processing (not needed with Qwen3) |

---

## Celebrity Voice Cloning Framework

A unified framework for cloning 22 iconic celebrity voices as an educational project exploring vocal diversity. Built on the same Qwen3-TTS Base ICL pipeline proven with Adam Barrow and Gregory Gorin.

### Quick Start

```bash
cd /Users/gregorygorin/Projects/voice/mlx-tts
source .venv/bin/activate

# List all available voices
python ../MOSS-TTS/generate_celebrity.py --list

# Generate a specific celebrity's monologue
python ../MOSS-TTS/generate_celebrity.py morgan_freeman --play

# Generate all 22 celebrities
python ../MOSS-TTS/generate_celebrity.py --all

# Custom text with any voice
python ../MOSS-TTS/generate_celebrity.py morgan_freeman --text "Your custom text here."
```

### The 22 Voices

| # | Celebrity | Speed | Mode |
|---|-----------|-------|------|
| 1 | Morgan Freeman | 0.82 | single_pass |
| 2 | David Attenborough | 0.85 | single_pass |
| 3 | Oprah Winfrey | 0.90 | single_pass |
| 4 | Joe Rogan | 0.95 | single_pass |
| 5 | Ira Glass | 0.88 | single_pass |
| 6 | Cate Blanchett | 0.88 | single_pass |
| 7 | Barack Obama | 0.85 | single_pass |
| 8 | Anthony Bourdain | 0.90 | single_pass |
| 9 | Zendaya | 0.92 | single_pass |
| 10 | Werner Herzog | 0.83 | single_pass |
| 11 | Bill Murray | 0.88 | single_pass |
| 12 | Steve Martin | 0.90 | single_pass |
| 13 | Martin Short | 0.93 | single_pass |
| 14 | John Candy | 0.88 | single_pass |
| 15 | Conan O'Brien | 0.94 | single_pass |
| 16 | James Earl Jones | 0.80 | single_pass |
| 17 | Christopher Walken | 0.85 | single_pass |
| 18 | Jack Nicholson | 0.88 | single_pass |
| 19 | Sam Elliott | 0.84 | single_pass |
| 20 | Gilbert Gottfried | 0.95 | single_pass |
| 21 | Jack Lemmon | 0.90 | single_pass |
| 22 | Matthew McConaughey | 0.86 | single_pass |

### Directory Structure

```
celebrities/
  configs/                    # YAML configs (speed, mode, ref audio path, ref text)
    morgan_freeman.yaml
    ...
  profiles/                   # Voice profile guides (tics, patterns, pacing)
    morgan-freeman-voice-profile.md
    ...
  scripts/                    # Monologue scripts (100-150 word in-character pieces)
    morgan_freeman_monologue.yaml
    ...
  audio/                      # Reference audio clips (24kHz mono WAV, 10-15s)
    reference_morgan_freeman.wav
    ...
```

### Adding a New Celebrity

1. **Extract reference audio** from YouTube or a local file:
   ```bash
   python tools/extract_reference.py \
     --url "https://youtube.com/watch?v=XXXXX" \
     --start 120.0 --end 135.0 \
     --celebrity new_celebrity --candidate 1
   ```

2. **Listen and select** the best candidate:
   ```bash
   python tools/extract_reference.py --celebrity new_celebrity --list
   python tools/extract_reference.py --celebrity new_celebrity --promote 1
   ```

3. **Create YAML config** at `celebrities/configs/new_celebrity.yaml`:
   ```yaml
   name: "New Celebrity"
   slug: "new_celebrity"
   ref_audio: "celebrities/audio/reference_new_celebrity.wav"
   ref_text: >
     Exact transcript of the reference audio clip.
   generation:
     speed: 0.88
     mode: "single_pass"
   ```

4. **Write voice profile** at `celebrities/profiles/new_celebrity-voice-profile.md`

5. **Write monologue script** at `celebrities/scripts/new_celebrity_monologue.yaml`

6. **Generate and test**:
   ```bash
   python generate_celebrity.py new_celebrity --play
   ```

### Reference Audio Extraction Tool

```bash
# From YouTube
python tools/extract_reference.py --url "https://youtube.com/..." --start 120 --end 135 --celebrity morgan_freeman --candidate 1

# From local file
python tools/extract_reference.py --file interview.mp4 --start 45 --end 60 --celebrity morgan_freeman --candidate 2

# List candidates
python tools/extract_reference.py --celebrity morgan_freeman --list

# Promote best candidate to final reference
python tools/extract_reference.py --celebrity morgan_freeman --promote 1
```

### Known Limitations

- **Voice switching in chunked mode**: Some voices break with chunking. Default to single_pass.
- **Accent reproduction**: Qwen3 may flatten strong accents (e.g., Herzog's German). Test early.
- **Older recordings**: Source material from the 1970s-80s (John Candy, Jack Lemmon) may need extra denoising.
- **Length limit**: Keep single-pass generations under ~100-150 words to avoid degradation.

---

## Legacy: MOSS-TTS Voice Cloning

<details>
<summary>Click to expand MOSS-TTS workflow</summary>

### Generate with CLI

```bash
python narration_farmers_choice.py --voice adam_barrow          # slide 1 only
python narration_farmers_choice.py --voice adam_barrow --full    # all slides
python narration_farmers_choice.py --voice all --full            # all voices
```

### Use in Custom Scripts

```python
import mps_compat  # noqa — auto-patches MPS backend
from transformers import AutoModel, AutoProcessor
from optimized_generate import patch_generate

model_path = "OpenMOSS-Team/MOSS-TTS"
ref_audio = "assets/audio/reference_adam_barrow.wav"

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)

model = AutoModel.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to(device)
model.eval()
patch_generate(model)

conversations = [[
    processor.build_user_message(
        text="So here's what's interesting...",
        reference=[ref_audio],
        instruction="Natural, conversational tone like you're walking a client through findings.",
    )
]]

batch = processor(conversations, mode="generation")
outputs = model.generate(
    input_ids=batch["input_ids"].to(device),
    attention_mask=batch["attention_mask"].to(device),
    max_new_tokens=6144,
    audio_temperature=1.7, audio_top_p=0.8, audio_top_k=25,
    audio_repetition_penalty=1.0,
)
messages = processor.decode(outputs)
audio = messages[0].audio_codes_list[0].detach().float().cpu().numpy().reshape(-1)
```

### Per-Slide Emotion Instructions

MOSS-TTS `build_user_message()` accepts an `instruction` param for emotional direction. Qwen3 ICL does not — embed emotion in the text itself.

| Slide | Instruction |
|-------|-------------|
| Hook | "Natural, conversational tone like you're walking a client through findings in a relaxed meeting." |
| Audit | "Friendly and direct, like explaining results to a client you already know." |
| Plan | "Upbeat and energetic like you're genuinely excited to share the game plan." |
| CTA | "Warm and easy-going, zero pressure. Like chatting with a friend about next steps." |

### Performance (M4 Pro, MPS)
- Model load: ~16–20s
- Generation: ~38–46s for ~23–28s of audio (RTF ~1.6x)
- Output: 24kHz mono WAV

### Why we moved away from MOSS-TTS
- 18.8 dB SNR (noisy codec artifacts)
- Post-processing (EQ, exciter, compression, saturation) all made it worse by amplifying the noise floor
- 9.3kHz resonance peak (+13dB prominence) from the model's codec
- Qwen3-TTS Base ICL produces 34+ dB SNR with no processing needed

</details>
