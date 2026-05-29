"""
PyTorch Forward Hooks for SPREAD Activation Steering.

Supports:
  - Fixed SPREAD hook (constant C_alpha)
  - Gated SPREAD hook (consistency-adaptive C_alpha)
  - Strategy-biased Unified hook
"""
import torch
import numpy as np
from collections import Counter
from .spread import riemannian_block_update


def make_fixed_hook(model, layer, calpha, recalc, T=10):
    """
    Fixed SPREAD hook: constant steering strength.

    Args:
        model: HuggingFace model
        layer: int, which transformer layer to hook
        calpha: float, steering strength
        recalc: int, recompute steering every N tokens
        T: int, Riemannian BCD iterations

    Returns:
        List of hook handles (embed_tokens + layer hook)
    """
    input_ids = None
    gen_cnt = [0]
    v_steer = [None]

    def id_hook(module, input, output):
        nonlocal input_ids
        n = input[0].detach().clone()
        input_ids = torch.cat((input_ids, n), dim=1) if input_ids is not None else n
        return output

    def steer_hook(module, input, output):
        nonlocal gen_cnt, v_steer
        gen_cnt[0] += 1
        if input_ids is None:
            return output

        out_t = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else ()

        if (gen_cnt[0] - 3) % recalc == 0 or gen_cnt[0] == 3:
            h = out_t[:, -1, :].detach().cpu().numpy().astype(np.float32)
            v_steer[0] = torch.tensor(
                riemannian_block_update(h, T=T, calpha_k=calpha),
                dtype=out_t.dtype, device=out_t.device,
            )

        if v_steer[0] is not None and v_steer[0].shape[0] == out_t.shape[0]:
            out_t[:, -1, :] += v_steer[0]

        return (out_t, *rest) if isinstance(output, tuple) else out_t

    # Qwen3.5 model structure: model.model.embed_tokens / model.model.layers[i]
    # For other models, adjust the path accordingly
    return [
        model.model.embed_tokens.register_forward_hook(id_hook),
        model.model.layers[layer].register_forward_hook(steer_hook),
    ]


def make_gated_hook(model, layer, base_calpha, recalc, T=10):
    """
    Consistency-gated SPREAD hook: adapts C_alpha based on
    inter-branch token agreement rate and hidden-state diversity.

    Returns:
        hooks: List of hook handles
        monitor_log: shared list, populated with
            [(token_pos, agreement_rate, diversity_score, effective_calpha), ...]
    """
    input_ids = None
    prompt_len = [0]
    gen_cnt = [0]
    v_steer = [None]
    monitor_log = []

    def id_hook(module, input, output):
        nonlocal input_ids, prompt_len
        n = input[0].detach().clone()
        if input_ids is None:
            input_ids = n
            prompt_len[0] = n.shape[1]
        else:
            input_ids = torch.cat((input_ids, n), dim=1)
        return output

    def steer_hook(module, input, output):
        nonlocal gen_cnt, v_steer, monitor_log
        gen_cnt[0] += 1
        if input_ids is None:
            return output

        out_t = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else ()
        N = out_t.shape[0]

        if (gen_cnt[0] - 5) % recalc == 0 or gen_cnt[0] == 5:
            # ── Monitor: agreement rate ──
            gen_ids = input_ids[:, prompt_len[0]:]
            window = min(recalc, gen_ids.shape[1])
            recent = gen_ids[:, -window:]
            agree_rates = []
            for p in range(recent.shape[1]):
                tokens = recent[:, p].tolist()
                agree_rates.append(max(Counter(tokens).values()) / N)
            mean_agree = np.mean(agree_rates) if agree_rates else 0.5

            # ── Monitor: diversity score ──
            hn = out_t[:, -1, :]
            hn_norm = hn / (hn.norm(dim=1, keepdim=True) + 1e-8)
            cos_sim = (hn_norm @ hn_norm.T).abs()
            cos_sim.fill_diagonal_(0)
            div_score = 1 - cos_sim.mean().item()

            # ── Adaptive C_alpha ──
            if mean_agree > 0.7 and div_score < 0.3:
                ec = base_calpha * 1.2   # stuck → boost
            elif mean_agree < 0.4 and div_score > 0.5:
                ec = base_calpha * 0.3   # already diverse → maintain
            else:
                ec = base_calpha * (0.5 + 0.5 * (1 - mean_agree))

            monitor_log.append((
                gen_cnt[0],
                round(mean_agree, 3),
                round(div_score, 3),
                round(ec, 4),
            ))

            # ── Apply steering ──
            h = out_t[:, -1, :].detach().cpu().numpy().astype(np.float32)
            v_steer[0] = torch.tensor(
                riemannian_block_update(h, T=T, calpha_k=ec),
                dtype=out_t.dtype, device=out_t.device,
            )

        if v_steer[0] is not None and v_steer[0].shape[0] == N:
            out_t[:, -1, :] += v_steer[0]

        return (out_t, *rest) if isinstance(output, tuple) else out_t

    return [
        model.model.embed_tokens.register_forward_hook(id_hook),
        model.model.layers[layer].register_forward_hook(steer_hook),
    ], monitor_log
