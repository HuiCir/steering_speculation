"""
Steering-Speculation: Combined Pipeline Orchestrator
=====================================================
Three-phase diversity decoding pipeline:

  Phase 1: GENERATE — Spread+Prompt, N branches, batch parallel
  Phase 2: SELECT   — DraftTarget, adaptive branch count
  Phase 3: DELIVER  — Swarm dispatch, ensemble or best-diverse

Usage:
    from src.orchestrator import Orchestrator
    orch = Orchestrator(model, tokenizer)
    result = orch.run(prompt)  # returns dict with responses, diversity, pipeline trace
"""
import time
from .generate import generate
from .select import (
    select_phase,
    select_diverse_branches,
    branch_count_summary,
)
from .deliver import deliver_branches
from .metrics import jaccard_diversity


class Orchestrator:
    """
    Combined Pipeline Orchestrator.

    Args:
        model: HuggingFace causal LM (e.g., Qwen3.5-2B)
        tokenizer: tokenizer
        config: dict, overrides default parameters
    """

    def __init__(self, model, tokenizer, config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = {
            "num_candidates": 4,
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "steer_layer": 8,
            "base_calpha": 0.3,
            "recalc_every": 20,
            "similarity_threshold": 0.5,
            "seed": 42,
            "deliver_mode": "ensemble",
        }
        if config:
            self.cfg.update(config)

    def run(self, prompt):
        """
        Run the combined pipeline on a single prompt.

        Returns:
            dict with keys:
                - responses: list of K text strings (K ≤ N)
                - diversity: float, Jaccard diversity
                - pipeline_trace: list of phase decisions
                - branch_count_summary: {early, mid, late} branch counts
                - deliver_result: delivery phase output
                - timing: {generate, select, deliver} seconds
                - prompt_triggers: int
        """
        timing = {}

        # ── Phase 1: GENERATE ──
        t0 = time.time()
        responses, gen_time, monitor_log, prompt_triggers = generate(
            self.model,
            self.tokenizer,
            prompt,
            num_candidates=self.cfg["num_candidates"],
            max_new_tokens=self.cfg["max_new_tokens"],
            temperature=self.cfg["temperature"],
            top_p=self.cfg["top_p"],
            steer_layer=self.cfg["steer_layer"],
            base_calpha=self.cfg["base_calpha"],
            recalc_every=self.cfg["recalc_every"],
            similarity_threshold=self.cfg["similarity_threshold"],
            seed=self.cfg["seed"],
        )
        timing["generate"] = gen_time

        # ── Phase 2: SELECT ──
        t0 = time.time()
        decisions = select_phase(monitor_log)

        # Use majority decision from late phase
        if decisions:
            n = len(decisions)
            late_decisions = decisions[2 * n // 3 :] if n >= 3 else decisions
            final_bc = max(set(d[1] for d in late_decisions), key=[d[1] for d in late_decisions].count)
        else:
            final_bc = self.cfg["num_candidates"]

        selected_idx = select_diverse_branches(responses, final_bc)
        selected_responses = [responses[i] for i in selected_idx]
        bc_summary = branch_count_summary(decisions)
        timing["select"] = time.time() - t0

        # ── Phase 3: DELIVER ──
        t0 = time.time()
        deliver_result = deliver_branches(
            selected_responses, mode=self.cfg["deliver_mode"]
        )
        timing["deliver"] = time.time() - t0

        return {
            "responses": selected_responses,
            "diversity": jaccard_diversity(selected_responses),
            "pipeline_trace": decisions,
            "branch_count_summary": bc_summary,
            "deliver_result": deliver_result,
            "timing": timing,
            "prompt_triggers": prompt_triggers,
            "total_branches_generated": len(responses),
            "total_branches_kept": len(selected_responses),
        }
