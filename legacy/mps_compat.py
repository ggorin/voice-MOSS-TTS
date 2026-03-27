"""MPS compatibility patches for PyTorch operations.

Import this module *before* loading any models to apply monkey-patches
that work around known PyTorch MPS backend bugs.

Also provides MPS-native sampling functions that avoid CPU fallbacks
for top-k, top-p, and multinomial operations.

Usage:
    import mps_compat  # noqa: F401  — patches are applied on import
"""
import torch
import torch.nn.functional as F
from typing import Optional

_patched = False


def _cpu_fallback(orig_fn, input, *args, **kwargs):
    """Try MPS first; on failure move to CPU, run, move back."""
    try:
        return orig_fn(input, *args, **kwargs)
    except RuntimeError:
        dev = input.device
        result = orig_fn(input.cpu(), *args, **kwargs)
        if isinstance(result, torch.Tensor):
            return result.to(dev)
        # tuple of tensors (topk, sort)
        return tuple(t.to(dev) for t in result)


def apply_patches():
    """Apply MPS monkey-patches. Safe to call multiple times (idempotent)."""
    global _patched
    if _patched or not torch.backends.mps.is_available():
        return
    _patched = True

    # 1. index_fill / index_fill_ crash on zero-element tensors
    _orig_index_fill_ = torch.Tensor.index_fill_
    _orig_index_fill = torch.Tensor.index_fill

    def _safe_index_fill_(self, dim, index, value):
        if self.numel() == 0:
            return self
        return _orig_index_fill_(self, dim, index, value)

    def _safe_index_fill(self, dim, index, value):
        if self.numel() == 0:
            return self.clone()
        return _orig_index_fill(self, dim, index, value)

    torch.Tensor.index_fill_ = _safe_index_fill_
    torch.Tensor.index_fill = _safe_index_fill

    # 2. torch.topk — try MPS, fall back to CPU on error
    _orig_topk = torch.topk

    def _safe_topk(input, k, dim=-1, largest=True, sorted=True, *, out=None):
        if not input.is_mps:
            return _orig_topk(input, k, dim=dim, largest=largest, sorted=sorted, out=out)
        return _cpu_fallback(_orig_topk, input, k, dim=dim, largest=largest, sorted=sorted)

    torch.topk = _safe_topk

    # 3. torch.multinomial — try MPS, fall back to CPU on error
    _orig_multinomial = torch.multinomial

    def _safe_multinomial(input, num_samples, replacement=False, *, generator=None):
        if not input.is_mps:
            return _orig_multinomial(input, num_samples, replacement=replacement, generator=generator)
        return _cpu_fallback(
            _orig_multinomial, input, num_samples, replacement=replacement, generator=generator
        )

    torch.multinomial = _safe_multinomial

    # 4. torch.Tensor.masked_fill — try MPS, fall back to CPU on error
    _orig_masked_fill = torch.Tensor.masked_fill

    def _safe_masked_fill(self, mask, value):
        try:
            return _orig_masked_fill(self, mask, value)
        except RuntimeError:
            return _orig_masked_fill(self.cpu(), mask.cpu(), value).to(self.device)

    torch.Tensor.masked_fill = _safe_masked_fill

    # 5. torch.sort — try MPS, fall back to CPU on error
    _orig_sort = torch.sort

    def _safe_sort(input, dim=-1, descending=False, stable=False, *, out=None):
        if not input.is_mps:
            return _orig_sort(input, dim=dim, descending=descending, stable=stable, out=out)
        return _cpu_fallback(_orig_sort, input, dim=dim, descending=descending, stable=stable)

    torch.sort = _safe_sort


# ── MPS-native sampling functions ───────────────────────────────────


def apply_top_k_mps(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Top-k filtering using kthvalue instead of topk (avoids CPU fallback on MPS).

    Args:
        logits: [N, V] float tensor of logits.
        top_k: Number of top values to keep.

    Returns:
        Logits with all but the top-k values set to -inf.
    """
    if logits.numel() == 0:
        return logits
    vocab_size = logits.size(-1)
    top_k = min(top_k, vocab_size)
    # kthvalue finds k-th smallest; we want (vocab_size - top_k + 1)-th smallest
    # which equals the top_k-th largest value — our threshold.
    kth_index = vocab_size - top_k  # 0-indexed position for kthvalue (1-indexed internally)
    # kthvalue is 1-indexed, so +1
    threshold, _ = torch.kthvalue(logits, kth_index + 1, dim=-1, keepdim=True)
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p_after_topk_mps(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Top-p (nucleus) filtering optimized for post-top-k logits.

    After top-k, most values are -inf. We extract only the finite values,
    sort those (~25 elements for audio, ~50 for text), and apply nucleus
    filtering. Even if the small sort falls back to CPU, transferring ~25
    floats is negligible vs sorting the full vocabulary.

    Args:
        logits: [N, V] float tensor, already top-k filtered (most values are -inf).
        top_p: Cumulative probability threshold.

    Returns:
        Logits with additional values masked to -inf to satisfy top-p constraint.
    """
    if logits.numel() == 0:
        return logits
    batch_size, vocab_size = logits.shape
    for i in range(batch_size):
        row = logits[i]
        finite_mask = torch.isfinite(row)
        finite_vals = row[finite_mask]
        if finite_vals.numel() <= 1:
            continue
        # Sort only the finite values (small tensor, ~25 elements)
        # Use float32 for softmax/cumsum precision
        sorted_vals, sorted_idx = torch.sort(finite_vals.float(), descending=True)
        probs = F.softmax(sorted_vals, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        # Remove tokens with cumulative probability above the threshold
        # Keep at least the first token
        remove_mask = cumprobs > top_p
        remove_mask[0] = False
        # Map back: get the original vocab indices of removed tokens
        finite_indices = torch.where(finite_mask)[0]
        removed_indices = finite_indices[sorted_idx[remove_mask]]
        row[removed_indices] = float("-inf")
    return logits


def gumbel_multinomial_mps(
    logits: torch.Tensor, num_samples: int = 1
) -> torch.Tensor:
    """Multinomial sampling via Gumbel-max trick (fully MPS-native).

    Instead of softmax → multinomial (which falls back to CPU on MPS),
    uses: argmax(logits + Gumbel_noise) where Gumbel = -log(-log(U)).
    This is mathematically equivalent to softmax → multinomial sampling.

    Works directly on raw logits — no softmax needed.
    Computes noise in float32 for numerical stability (important for bfloat16 logits).

    Args:
        logits: [N, V] float tensor of (optionally filtered) logits.
        num_samples: Number of samples to draw (typically 1).

    Returns:
        [N, num_samples] long tensor of sampled token indices.
    """
    if logits.numel() == 0:
        return torch.empty(
            (*logits.shape[:-1], num_samples), dtype=torch.long, device=logits.device
        )
    # Compute in float32 for numerical stability (bfloat16 has only ~3 digits precision)
    logits_f32 = logits.float()
    if num_samples == 1:
        u = torch.rand_like(logits_f32).clamp_(min=1e-10, max=1.0 - 1e-7)
        gumbel_noise = -torch.log(-torch.log(u))
        return torch.argmax(logits_f32 + gumbel_noise, dim=-1, keepdim=True)
    results = []
    for _ in range(num_samples):
        u = torch.rand_like(logits_f32).clamp_(min=1e-10, max=1.0 - 1e-7)
        gumbel_noise = -torch.log(-torch.log(u))
        results.append(torch.argmax(logits_f32 + gumbel_noise, dim=-1, keepdim=True))
    return torch.cat(results, dim=-1)


def sample_token_mps(
    logits: torch.Tensor,
    prev_tokens: Optional[torch.LongTensor] = None,
    repetition_penalty: float = 1.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    do_sample: bool = True,
    _apply_repetition_penalty=None,
    _apply_top_p_optimized=None,
) -> torch.Tensor:
    """Drop-in replacement for inference_utils.sample_token using MPS-native ops.

    Eliminates CPU fallbacks for topk and multinomial by using:
    - kthvalue-based top-k (instead of torch.topk)
    - Gumbel-max trick for multinomial (instead of torch.multinomial)
    Uses original vectorized top-p (which sorts the full vocab via CPU fallback
    but is faster than the Python-loop alternative).

    Args:
        logits: Token logits, same shapes as original sample_token.
        prev_tokens: Previous tokens for repetition penalty.
        repetition_penalty: Repetition penalty factor.
        top_p: Nucleus sampling threshold.
        top_k: Top-k filtering count.
        do_sample: Whether to sample (True) or take argmax (False).
        _apply_repetition_penalty: The repetition penalty function to use.
        _apply_top_p_optimized: The original vectorized top-p function.

    Returns:
        Sampled token indices with same shape semantics as original.
    """
    vocab_size = logits.size(-1)

    if prev_tokens is not None and repetition_penalty != 1.0 and _apply_repetition_penalty is not None:
        logits = _apply_repetition_penalty(logits, prev_tokens, repetition_penalty)

    if not do_sample:
        return torch.argmax(logits, dim=-1)

    original_shape = logits.shape
    reshaped_logits = logits.view(-1, vocab_size)

    if top_k is not None and top_k > 0:
        reshaped_logits = apply_top_k_mps(reshaped_logits, top_k)

    if top_p is not None and top_p < 1.0:
        if _apply_top_p_optimized is not None:
            reshaped_logits = _apply_top_p_optimized(reshaped_logits, top_p)
        else:
            reshaped_logits = apply_top_p_after_topk_mps(reshaped_logits, top_p)

    next_tokens = gumbel_multinomial_mps(reshaped_logits, num_samples=1)

    return next_tokens.view(original_shape[:-1])


# Auto-apply on import
apply_patches()
