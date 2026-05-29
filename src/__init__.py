"""
Steering-Speculation: Combined Pipeline for Diverse Multi-Branch Decoding.

Phases:
  1. GENERATE — Spread+Prompt (activation steering + text fallback)
  2. SELECT   — DraftTarget (adaptive branch count from agreement/diversity)
  3. DELIVER  — Swarm dispatch (ensemble, best-diverse, or single)
"""

from .orchestrator import Orchestrator
from .spread import riemannian_block_update
from .generate import generate
from .select import select_phase, select_diverse_branches, decide_branch_count
from .deliver import deliver_branches, classify_branch_strategy
from .metrics import jaccard_diversity, rouge_l_score
from .hooks import make_fixed_hook, make_gated_hook
from .utils import load_sample, discover_samples

__version__ = "0.1.0"
__all__ = [
    "Orchestrator",
    "riemannian_block_update",
    "generate",
    "select_phase",
    "select_diverse_branches",
    "decide_branch_count",
    "deliver_branches",
    "classify_branch_strategy",
    "jaccard_diversity",
    "rouge_l_score",
    "make_fixed_hook",
    "make_gated_hook",
    "load_sample",
    "discover_samples",
]
