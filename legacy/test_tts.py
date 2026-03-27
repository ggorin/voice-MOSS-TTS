"""Minimal MOSS-TTS test script for macOS (MPS/CPU)."""

import time
import torch
import numpy as np
import soundfile as sf

import mps_compat  # noqa: F401 — apply MPS patches before model import

from transformers import AutoModel, AutoProcessor

from optimized_generate import patch_generate

MODEL_PATH = "OpenMOSS-Team/MOSS-TTS"

# Select device and dtype
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    attn_impl = "flash_attention_2"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype = torch.bfloat16
    attn_impl = "sdpa"
else:
    device = torch.device("cpu")
    dtype = torch.float32
    attn_impl = "eager"

print(f"Using device: {device}, dtype: {dtype}, attn: {attn_impl}")

# Load model and processor
print("Loading processor...")
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
if hasattr(processor, "audio_tokenizer"):
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)
print(f"Processor loaded in {time.time() - t0:.1f}s")

print("Loading model...")
t0 = time.time()
model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=dtype,
    attn_implementation=attn_impl,
).to(device)
model.eval()
patch_generate(model)
print(f"Model loaded in {time.time() - t0:.1f}s")

sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))

# Generate speech
text = (
    "Hello! This is a test of the MOSS text to speech system running on Apple Silicon."
)
print(f"\nGenerating speech for: {text!r}")

conversations = [[processor.build_user_message(text=text)]]
batch = processor(conversations, mode="generation")
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)

t0 = time.time()
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=4096,
        audio_temperature=1.7,
        audio_top_p=0.8,
        audio_top_k=25,
        audio_repetition_penalty=1.0,
    )
elapsed = time.time() - t0
print(f"Generation took {elapsed:.2f}s")

messages = processor.decode(outputs)
audio = messages[0].audio_codes_list[0]
if isinstance(audio, torch.Tensor):
    audio_np = audio.detach().float().cpu().numpy()
else:
    audio_np = np.asarray(audio, dtype=np.float32)
if audio_np.ndim > 1:
    audio_np = audio_np.reshape(-1)

# Save output
output_path = "test_output.wav"
sf.write(output_path, audio_np, sample_rate)
duration = len(audio_np) / sample_rate
print(f"\nSaved {output_path} ({duration:.2f}s audio, {sample_rate}Hz)")
print("Done!")
