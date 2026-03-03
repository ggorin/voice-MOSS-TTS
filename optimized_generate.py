"""Optimized generate loop for MOSS-TTS on MPS.

Monkey-patches ``MossTTSDelayModel.generate`` and ``forward`` with versions that:
 - replaces O(n^2) torch.cat growth with pre-allocated buffers
 - reduces GPU->CPU sync points (.any()/.all() + batched stopping check)
 - eliminates per-step tensor allocations for reused buffers
 - removes redundant .clone() under inference_mode
 - uses MPS-native sampling (kthvalue top-k, Gumbel-max multinomial)
   to eliminate ~900 CPU round-trips per generation
 - skips the text lm_head matmul (~637M multiply-adds) during audio-only steps
 - disables output_hidden_states when hidden_out_layers is not used

Usage::

    from optimized_generate import patch_generate
    patch_generate(model)       # call once after model load
    outputs = model.generate(...)  # now uses the fast path
"""

import functools
import os
import sys
import types

import torch
from tqdm import tqdm

from mps_compat import sample_token_mps


class _SkippableLinear(torch.nn.Module):
    """Wraps a Linear layer so it can be conditionally skipped (returns zeros)."""

    def __init__(self, orig_linear):
        super().__init__()
        self._orig = orig_linear
        self.skip = False
        self.out_features = orig_linear.out_features

    def forward(self, x):
        if self.skip:
            return torch.zeros(
                *x.shape[:-1], self.out_features, dtype=x.dtype, device=x.device
            )
        return self._orig(x)


def _patch_forward(model):
    """Monkey-patch model.forward to skip text lm_head and disable hidden state storage.

    Two optimizations applied via thin wrappers:
    1. Wrap language_model.forward to pass output_hidden_states=False during generation
    2. Wrap lm_heads[0] with a skippable wrapper so the text matmul can be skipped
    """
    _orig_forward = model.forward.__func__

    # 1. Wrap lm_heads[0] for conditional skip
    _text_head_wrapper = _SkippableLinear(model.lm_heads[0])
    model.lm_heads[0] = _text_head_wrapper

    # 2. Patch language_model.forward for hidden-state suppression
    _orig_lm_forward = model.language_model.forward

    def _lm_forward_no_hidden(*args, **kwargs):
        if getattr(model, "_force_no_hidden_states", False):
            kwargs["output_hidden_states"] = False
        return _orig_lm_forward(*args, **kwargs)

    model.language_model.forward = _lm_forward_no_hidden

    @functools.wraps(_orig_forward)
    def _patched_forward(self, *args, _skip_text_head=False, **kwargs):
        suppress_hidden = (
            kwargs.get("hidden_out_layers") is None and "labels" not in kwargs
        )
        self._force_no_hidden_states = suppress_hidden
        _text_head_wrapper.skip = _skip_text_head
        try:
            return _orig_forward(self, *args, **kwargs)
        finally:
            self._force_no_hidden_states = False
            _text_head_wrapper.skip = False

    model.forward = types.MethodType(_patched_forward, model)


def patch_generate(model):
    """Replace *model*.generate with an optimized pre-allocated-buffer version."""

    # Grab inference helpers from the model's own package so we stay in sync
    # with whichever revision was downloaded from the Hub.
    _mod = sys.modules[type(model).__module__]
    _sample_token_orig = _mod.sample_token
    _find_last_equal_C = _mod.find_last_equal_C
    # apply_repetition_penalty_delay_pattern and apply_top_p_optimized live in .inference_utils
    _inference_utils_mod = sys.modules[_mod.__name__.rsplit(".", 1)[0] + ".inference_utils"]
    _apply_rep_penalty = _inference_utils_mod.apply_repetition_penalty_delay_pattern
    _apply_top_p = _inference_utils_mod.apply_top_p_optimized

    # Detect MPS device (can be disabled via env var for debugging)
    _is_mps = (
        next(model.parameters()).device.type == "mps"
        and not os.environ.get("MOSS_NO_MPS_SAMPLING")
    )

    def _sample_fn(*, logits, prev_tokens=None, repetition_penalty=1.0,
                   top_p=None, top_k=None, do_sample=True):
        if _is_mps:
            return sample_token_mps(
                logits=logits,
                prev_tokens=prev_tokens,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                _apply_repetition_penalty=_apply_rep_penalty,
                _apply_top_p_optimized=_apply_top_p,
            )
        return _sample_token_orig(
            logits=logits,
            prev_tokens=prev_tokens,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
        )

    # Patch forward for hidden state / text head optimizations
    _patch_forward(model)

    @torch.inference_mode()
    def _optimized_generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask=None,
        max_new_tokens: int = 1000,
        text_temperature: float = 1.5,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 1.7,
        audio_top_p: float = 0.8,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.0,
    ):
        # ── temperature -> do_sample flags ────────────────────────────
        if text_temperature > 0:
            text_do_sample = True
        else:
            text_temperature = 1
            text_do_sample = False
        if audio_temperature > 0:
            audio_do_sample = True
        else:
            audio_temperature = 1
            audio_do_sample = False

        device = input_ids.device
        batch_size, seq_len, n_vq_plus1 = input_ids.shape
        n_vq = n_vq_plus1 - 1
        total_len = seq_len + max_new_tokens

        # ── Pre-allocate buffers (replaces O(n^2) torch.cat growth) ──
        generation_ids = torch.full(
            (batch_size, total_len, n_vq_plus1),
            self.config.audio_pad_code,
            dtype=input_ids.dtype,
            device=device,
        )
        generation_ids[:, :seq_len, :] = input_ids
        gen_len = seq_len

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                seq_len,
                dtype=torch.int64,
                device=device,
            )
        attn_mask_buf = torch.zeros(
            batch_size,
            total_len,
            dtype=attention_mask.dtype,
            device=device,
        )
        attn_mask_buf[:, :seq_len] = attention_mask
        current_attention_mask = attn_mask_buf[:, :seq_len]

        # Reusable per-step token buffers (avoids per-step torch.full)
        next_text_token = torch.full(
            (batch_size,),
            self.config.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        next_audio_tokens = torch.full(
            (batch_size, n_vq),
            self.config.audio_pad_code,
            dtype=torch.long,
            device=device,
        )

        # Pre-compute arange reused every step for mask construction
        vq_range = torch.arange(n_vq, dtype=torch.int64, device=device).expand(
            batch_size,
            n_vq,
        )

        # ── Tracking state (identical logic to original) ─────────────
        past_key_values = None
        current_input_ids = input_ids

        is_stopping = torch.zeros(batch_size, dtype=torch.bool, device=device)
        audio_lengths = torch.zeros(batch_size, dtype=torch.int64, device=device)
        torch_int64_max = torch.iinfo(torch.int64).max
        delayed_lengths = torch.full(
            (batch_size,),
            torch_int64_max,
            dtype=torch.int64,
            device=device,
        )

        is_continuation = (input_ids[:, -1, 0] == self.config.audio_start_token_id) | (
            input_ids[:, -1, 0] == self.config.audio_assistant_gen_slot_token_id
        )
        audio_start_indices = _find_last_equal_C(
            input_ids[..., 0],
            self.config.audio_start_token_id,
        )
        audio_start_mask = is_continuation & (audio_start_indices != -1)
        audio_lengths[audio_start_mask] = (
            seq_len - audio_start_indices[audio_start_mask]
        )
        is_audio = audio_start_mask.clone()

        pre_exclude_mask0 = torch.tensor(
            [
                self.config.pad_token_id,
                self.config.audio_assistant_gen_slot_token_id,
                self.config.audio_assistant_delay_slot_token_id,
                self.config.audio_end_token_id,
            ],
            device=device,
        )
        pre_exclude_mask1 = torch.ones(
            self.config.language_config.vocab_size,
            device=device,
        ).bool()
        pre_exclude_mask1[
            [
                self.config.audio_assistant_gen_slot_token_id,
                self.config.audio_assistant_delay_slot_token_id,
            ]
        ] = False

        # Track whether text logits will be needed on the NEXT step.
        # Updated at end of each step using state that's already synced.
        # Avoids an extra GPU→CPU sync per step.
        _text_needed_next = True  # Start True (first step always needs text logits)

        # ── Generate loop ─────────────────────────────────────────────
        for time_step in tqdm(
            range(max_new_tokens), desc=f"Generating bs{batch_size} ..."
        ):
            skip_text = _is_mps and not _text_needed_next

            outputs = self(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                _skip_text_head=skip_text,
            )
            past_key_values = outputs.past_key_values

            # Temperature-scaled last-token logits.
            # Division already produces new tensors; .clone() is unnecessary
            # under @torch.inference_mode().
            next_token_logits = [
                logit[:, -1, :] / text_temperature
                if idx == 0
                else logit[:, -1, :] / audio_temperature
                for idx, logit in enumerate(outputs.logits)
            ]

            # ── Text token decision ──────────────────────────────────
            next_text_token.fill_(self.config.pad_token_id)
            next_text_token[~is_stopping & (delayed_lengths < n_vq)] = (
                self.config.audio_assistant_delay_slot_token_id
            )
            is_audio_eos = ~is_stopping & (delayed_lengths == n_vq)
            next_text_token[is_audio_eos] = self.config.audio_end_token_id
            is_audio[is_audio_eos] = False
            sampling_text_mask = ~is_stopping & (delayed_lengths > n_vq)

            next_token_logits[0][~is_audio] = next_token_logits[0][
                ~is_audio
            ].index_fill(
                -1,
                pre_exclude_mask0,
                float("-inf"),
            )
            next_token_logits[0][is_audio] = next_token_logits[0][is_audio].masked_fill(
                pre_exclude_mask1,
                float("-inf"),
            )
            if time_step == 0:
                next_token_logits[0][..., 151662] = float("-inf")
            if time_step <= n_vq:
                next_token_logits[0][..., self.config.im_end_token_id] = float("-inf")

            next_text_token[sampling_text_mask] = _sample_fn(
                logits=next_token_logits[0][sampling_text_mask],
                top_p=text_top_p,
                top_k=text_top_k,
                do_sample=text_do_sample,
            )
            is_audio[next_text_token == self.config.audio_start_token_id] = True
            is_stopping[next_text_token == self.config.im_end_token_id] = True

            # ── Audio token decision ─────────────────────────────────
            next_audio_tokens.fill_(self.config.audio_pad_code)

            pre_audio_mask = audio_lengths.unsqueeze(1) > vq_range
            post_audio_mask = vq_range > delayed_lengths.unsqueeze(1) - 1
            post_audio_mask[delayed_lengths == torch_int64_max] = True
            sampling_audio_mask = pre_audio_mask & post_audio_mask

            # .any() avoids the full .sum() GPU->CPU sync
            if sampling_audio_mask.any():
                gen_view = generation_ids[:, :gen_len, :]
                audio_ch0_logits = next_token_logits[1][sampling_audio_mask[:, 0]]
                audio_logits = torch.stack(next_token_logits[2:], dim=1)[
                    sampling_audio_mask[:, 1:]
                ]
                audio_ch0_logits[..., self.config.audio_pad_code] = float("-inf")
                audio_logits[..., self.config.audio_pad_code] = float("-inf")
                next_audio_tokens[:, 0][sampling_audio_mask[:, 0]] = _sample_fn(
                    logits=audio_ch0_logits,
                    prev_tokens=gen_view[:, :, 1],
                    repetition_penalty=audio_repetition_penalty,
                    top_p=audio_top_p,
                    top_k=audio_top_k,
                    do_sample=audio_do_sample,
                )
                next_audio_tokens[:, 1:][sampling_audio_mask[:, 1:]] = _sample_fn(
                    logits=audio_logits,
                    prev_tokens=gen_view[:, :, 2:],
                    repetition_penalty=audio_repetition_penalty,
                    top_p=audio_top_p,
                    top_k=audio_top_k,
                    do_sample=audio_do_sample,
                )

            # ── Update tracking state ────────────────────────────────
            audio_lengths[
                (next_text_token == self.config.audio_start_token_id)
                | (next_text_token == self.config.audio_assistant_gen_slot_token_id)
                | (next_text_token == self.config.audio_assistant_delay_slot_token_id)
            ] += 1
            audio_lengths[next_text_token == self.config.audio_end_token_id] = 0
            delayed_lengths[
                (delayed_lengths == torch_int64_max)
                & (next_text_token == self.config.audio_assistant_delay_slot_token_id)
            ] = 0
            delayed_lengths[delayed_lengths != torch_int64_max] += 1
            delayed_lengths[delayed_lengths > n_vq] = torch_int64_max

            # ── Write step into pre-allocated buffers (O(1) vs O(n)) ─
            generation_ids[:, gen_len, 0] = next_text_token
            generation_ids[:, gen_len, 1:] = next_audio_tokens
            attn_mask_buf[:, gen_len] = (~is_stopping).to(attn_mask_buf.dtype)

            # View of the token we just wrote feeds the next forward pass
            current_input_ids = generation_ids[:, gen_len : gen_len + 1, :]
            gen_len += 1
            current_attention_mask = attn_mask_buf[:, :gen_len]

            # Update text-needed prediction for next step.
            # After delayed_lengths is updated: if any element has delayed_lengths > n_vq
            # (which was just reset to int64_max), text logits will be needed.
            # For batch_size=1, track via Python to avoid GPU→CPU sync.
            if batch_size == 1:
                # delayed_lengths was just updated. If it's int64_max, text is needed.
                # We can infer this from the token we just generated:
                # - delay/end tokens → delayed_lengths in [0, n_vq] → text NOT needed
                # - all other tokens (pad, start, gen, im_end) → int64_max → text needed
                t = next_text_token[0].item()  # already synced for stopping check
                # Only delay_slot means text is not needed next step.
                # audio_end resets delayed_lengths to int64_max (via n_vq→n_vq+1→max),
                # so text IS needed after audio_end.
                _text_needed_next = (
                    t != self.config.audio_assistant_delay_slot_token_id
                )
            else:
                _text_needed_next = bool(
                    (~is_stopping & (delayed_lengths > n_vq)).any()
                )

            # Batched stopping check every 4 steps to reduce GPU->CPU syncs
            if time_step % 4 == 3:
                if is_stopping.all():
                    break

        # ── Trim and build output ─────────────────────────────────────
        generation_ids = generation_ids[:, :gen_len, :]

        start_indices = (
            _find_last_equal_C(input_ids[..., 0], self.config.im_start_token_id) + 3
        )
        start_lengths = seq_len - start_indices

        output = []
        for start_idx, start_length, cur_gen_ids in zip(
            start_indices,
            start_lengths,
            generation_ids,
        ):
            output.append((start_length, cur_gen_ids[start_idx:]))

        return output

    model.generate = types.MethodType(_optimized_generate, model)
    opts = []
    if _is_mps:
        opts.append("MPS-native sampling")
        opts.append("text-head skip")
    opts.append("hidden-state optimization")
    opts.append("pre-allocated buffers")
    print(f"[optimized_generate] Patched model with: {', '.join(opts)}")
