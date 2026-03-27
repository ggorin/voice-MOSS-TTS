"""Three-way benchmark: original vs buffer-only vs fully-optimized generate."""
import time

import torch
import numpy as np

import mps_compat  # noqa: F401

from transformers import AutoModel, AutoProcessor

MODEL_PATH = "OpenMOSS-Team/MOSS-TTS"
NUM_RUNS = 3

if torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype = torch.bfloat16
    attn_impl = "sdpa"
else:
    device = torch.device("cpu")
    dtype = torch.float32
    attn_impl = "eager"

print(f"Device: {device}, dtype: {dtype}, attn: {attn_impl}")

# Load model
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
if hasattr(processor, "audio_tokenizer"):
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)

model = AutoModel.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype=dtype, attn_implementation=attn_impl,
).to(device)
model.eval()

# Save the original generate and forward methods
original_generate = model.generate
original_forward = model.forward

# Prepare input
text = "Hello! This is a test of the MOSS text to speech system running on Apple Silicon."
conversations = [[processor.build_user_message(text=text)]]
batch = processor(conversations, mode="generation")
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)

gen_kwargs = dict(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=4096,
    audio_temperature=1.7,
    audio_top_p=0.8,
    audio_top_k=25,
    audio_repetition_penalty=1.0,
)

sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))


def run_once(label):
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    messages = processor.decode(outputs)
    audio = messages[0].audio_codes_list[0]
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().float().cpu().numpy()
    else:
        audio_np = np.asarray(audio, dtype=np.float32)
    if audio_np.ndim > 1:
        audio_np = audio_np.reshape(-1)
    duration = len(audio_np) / sample_rate
    rtf = elapsed / duration if duration > 0 else float("inf")
    print(f"  [{label}] {elapsed:.2f}s wall, {duration:.2f}s audio, RTF={rtf:.2f}x")
    return elapsed, duration


def restore_original():
    """Restore model to its original unpatched state."""
    model.generate = original_generate
    model.forward = original_forward


# ── Warmup ──
print("\nWarmup run (discarded)...")
restore_original()
run_once("warmup-original")

from optimized_generate import patch_generate  # noqa: E402
patch_generate(model)
run_once("warmup-optimized")

# ── Benchmark: Original ──
print(f"\n{'='*60}")
print(f"Original generate ({NUM_RUNS} runs)")
print("  No optimizations -- stock HuggingFace generate")
print(f"{'='*60}")
orig_times = []
orig_durations = []
for i in range(NUM_RUNS):
    restore_original()
    elapsed, duration = run_once(f"run {i+1}")
    orig_times.append(elapsed)
    orig_durations.append(duration)

# ── Benchmark: Fully Optimized ──
print(f"\n{'='*60}")
print(f"Fully optimized ({NUM_RUNS} runs)")
print("  MPS-native sampling + text-head skip + hidden-state opt + buffers")
print(f"{'='*60}")
opt_times = []
opt_durations = []
for i in range(NUM_RUNS):
    restore_original()
    patch_generate(model)
    elapsed, duration = run_once(f"run {i+1}")
    opt_times.append(elapsed)
    opt_durations.append(duration)

# ── Results ──
orig_avg = np.mean(orig_times)
orig_dur_avg = np.mean(orig_durations)
opt_avg = np.mean(opt_times)
opt_dur_avg = np.mean(opt_durations)
speedup_pct = (orig_avg - opt_avg) / orig_avg * 100
time_saved = orig_avg - opt_avg

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Original:        {orig_avg:.2f}s avg ({orig_dur_avg:.2f}s audio, RTF={orig_avg/orig_dur_avg:.2f}x)")
print(f"Fully optimized: {opt_avg:.2f}s avg ({opt_dur_avg:.2f}s audio, RTF={opt_avg/opt_dur_avg:.2f}x)")
print(f"Speedup: {speedup_pct:+.1f}% ({time_saved:.2f}s saved)")
