"""
Phase 1: Generate — Diversity Maximization
==========================================
Uses batch generation + SPREAD hooks + prompt fallback.

The Generate phase always runs at maximum parallelism (N branches).
It produces N candidate responses plus a monitor log that feeds
into the Select phase.
"""
import time
import re
import torch
from .hooks import make_gated_hook
from .metrics import token_jaccard_similarity


def _prepare_input(tokenizer, prompt, device):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt").to(device)


def generate(
    model,
    tokenizer,
    prompt,
    num_candidates=4,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    steer_layer=8,
    base_calpha=0.3,
    recalc_every=20,
    similarity_threshold=0.5,
    seed=42,
):
    """
    Phase 1 — Generate diverse candidates with SPREAD + prompt fallback.

    Args:
        model: HuggingFace causal LM
        tokenizer: tokenizer
        prompt: text prompt
        num_candidates: N, number of parallel branches
        max_new_tokens: max tokens to generate
        temperature: sampling temperature
        top_p: nucleus sampling p
        steer_layer: which transformer layer to hook
        base_calpha: base steering strength
        recalc_every: recompute steering every N tokens
        similarity_threshold: Jaccard similarity above which prompt fallback triggers
        seed: random seed

    Returns:
        responses: list of N strings
        elapsed: float, wall-clock seconds
        monitor_log: list of (pos, agree_rate, div_score, c_alpha)
        prompt_triggers: int, number of prompt fallback invocations
    """
    torch.manual_seed(seed)

    hooks, monitor_log = make_gated_hook(
        model, steer_layer, base_calpha, recalc_every
    )

    inp = _prepare_input(tokenizer, prompt, model.device)
    input_len = inp["input_ids"].shape[1]

    torch.cuda.synchronize()
    t0 = time.time()

    try:
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_candidates,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        for h in hooks:
            h.remove()

    torch.cuda.synchronize()
    elapsed = time.time() - t0

    responses = [
        tokenizer.decode(out[i][input_len:], skip_special_tokens=True)
        for i in range(out.shape[0])
    ]
    del out, inp
    torch.cuda.empty_cache()

    # ── Prompt Fallback ──
    prompt_triggers = 0
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            sim = token_jaccard_similarity(responses[i], responses[j])
            if sim > similarity_threshold:
                prompt_triggers += 1
                diverge_prompt = (
                    f"[DIVERGE from: {responses[i][:200]}]\n"
                    f"Generate a DIFFERENT analysis approach.\n\n{prompt}"
                )
                inp2 = _prepare_input(tokenizer, diverge_prompt, model.device)
                ilen2 = inp2["input_ids"].shape[1]
                with torch.no_grad():
                    out2 = model.generate(
                        **inp2,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.8,
                        top_p=top_p,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                responses[j] = tokenizer.decode(
                    out2[0][ilen2:], skip_special_tokens=True
                )
                del out2, inp2
                torch.cuda.empty_cache()

    return responses, elapsed, monitor_log, prompt_triggers
