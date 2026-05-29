# Steering-Speculation

## Combined Pipeline for Diverse Multi-Branch Speculative Decoding

**Generate(Spread+Prompt) → Select(DraftTarget) → Deliver(Swarm)**

A three-phase decoding pipeline that produces diverse reasoning branches in a single batch forward pass, adaptively selects the optimal number of branches, and dispatches results to downstream agent swarms.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  ORCHESTRATOR                             │
│                                                           │
│  PHASE 1: DRAFT GENERATE                                  │
│  ├─ Batch N=4 with SPREAD activation steering             │
│  ├─ Continuous monitoring: agreement_rate, diversity      │
│  └─ Prompt fallback on high branch similarity             │
│                      ↓                                    │
│  PHASE 2: TARGET SELECT                                   │
│  ├─ Adaptive branch count: clamp(N, agree, div)           │
│  ├─ Constraint: high agree → ↓branches (no waste)         │
│  ├─ Constraint: high entropy → ↑branches (no collapse)    │
│  └─ Greedy max-diversity subset selection                 │
│                      ↓                                    │
│  PHASE 3: DELIVER                                         │
│  ├─ ensemble → full swarm dispatch                        │
│  ├─ best_diverse → top-2 most distinct strategies         │
│  └─ single → single answer with confidence                │
└──────────────────────────────────────────────────────────┘
```

### Installation

```bash
pip install -r requirements.txt
```

Requires PyTorch 2.0+, Transformers 4.51+, and a compatible GPU with 8GB+ VRAM.

### Quick Start

```bash
# Run demo on 10 sample cases
python scripts/run_demo.py --model /path/to/qwen3.5-2b

# Run full benchmark (Batch vs SPREAD vs Spread+Prompt vs COMBINED)
python scripts/run_benchmark.py --model /path/to/qwen3.5-2b
```

### Python API

```python
from src import Orchestrator
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "path/to/qwen3.5-2b",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("path/to/qwen3.5-2b", trust_remote_code=True)

# Run pipeline
orch = Orchestrator(model, tokenizer, config={
    "num_candidates": 4,
    "deliver_mode": "ensemble",  # or "best_diverse", "single"
})
result = orch.run("Your prompt here")

print(f"Diversity: {result['diversity']}")
print(f"Branches: {result['total_branches_generated']} → {result['total_branches_kept']}")
print(f"Pipeline trace: {len(result['pipeline_trace'])} checkpoints")
```

### Sample Cases

The `samples/` directory contains 10 annotated cases from trajectory-bench:

| Type | Case | Tools | Domain |
|------|------|-------|--------|
| Parallel | P-Travel | 3 | Travel booking |
| Parallel | P-Finance | 3 | Stock analysis |
| Parallel | P-Email | 3 | Email management |
| Parallel | P-Gaming | 3 | Game data |
| Parallel | P-Weather | 3 | Weather API |
| Sequential | S-Travel | 3 | Hotel search chain |
| Sequential | S-Finance | 3 | Technical analysis |
| Sequential | S-Education | 3 | Course search |
| Sequential | S-News | 10 | News aggregation |
| Sequential | S-eCommerce | 3 | Product search |

Each sample includes: `query`, `tool_list` (with executed outputs), and `reference` (ground-truth answer for evaluation).

### Configuration

Edit `config/default.yaml` or pass a config dict to Orchestrator:

```yaml
num_candidates: 4        # N, parallel branches
steer_layer: 8            # Transformer layer for SPREAD hook
base_calpha: 0.3          # Steering strength
recalc_every: 20          # Recompute steering every N tokens
similarity_threshold: 0.5 # Trigger prompt fallback
deliver_mode: ensemble    # ensemble | best_diverse | single
```

