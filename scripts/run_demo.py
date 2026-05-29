"""
Steering-Speculation Demo Script
=================================
Runs the combined pipeline on the 10 sample cases.
"""
import sys
import time
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import Orchestrator, load_sample, discover_samples, jaccard_diversity, rouge_l_score


def main():
    parser = argparse.ArgumentParser(description="Steering-Speculation Demo")
    parser.add_argument("--model", required=True,
                        help="Path to HuggingFace model (e.g., /path/to/qwen3.5-2b)")
    parser.add_argument("--samples", default="samples",
                        help="Path to samples directory")
    parser.add_argument("--deliver", default="ensemble",
                        choices=["ensemble", "best_diverse", "single"])
    parser.add_argument("--max-cases", type=int, default=10,
                        help="Max number of cases to run")
    args = parser.parse_args()

    print("=" * 70)
    print("Steering-Speculation: Combined Pipeline Demo")
    print("=" * 70)

    # Load model
    print(f"\n[Load] {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    print(f"[Load] {sum(p.numel() for p in model.parameters())/1e9:.2f}B params, "
          f"{time.time()-t0:.1f}s")
    print(f"[GPU]  {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB VRAM")

    # Setup orchestrator
    orch = Orchestrator(model, tokenizer, config={
        "deliver_mode": args.deliver,
        "max_new_tokens": 200,
    })

    # Discover samples
    samples = discover_samples(args.samples)
    print(f"\n[Samples] {len(samples)} cases found")
    samples = samples[: args.max_cases]

    # Run pipeline
    results = []
    for name, filepath in samples:
        case = load_sample(filepath)
        print(f"\n{'─'*60}")
        print(f"[{name}] {case['tool_count']} tools | query: {case['query'][:80]}...")

        result = orch.run(case["prompt"])
        rl = rouge_l_score(result["responses"][0], case["reference"])

        print(f"  Generated {result['total_branches_generated']} branches, "
              f"kept {result['total_branches_kept']}")
        print(f"  Diversity Jaccard: {result['diversity']}")
        print(f"  ROUGE-L (best vs ref): {rl:.3f}")
        print(f"  Prompt fallbacks: {result['prompt_triggers']}")
        print(f"  BC summary: {result['branch_count_summary']}")
        print(f"  Timing: gen={result['timing']['generate']:.1f}s, "
              f"sel={result['timing']['select']:.3f}s, "
              f"del={result['timing']['deliver']:.3f}s")
        print(f"  Delivery: {result['deliver_result']['mode']} — "
              f"{result['deliver_result']['recommendation'][:80]}")

        results.append({
            "name": name,
            "diversity": result["diversity"],
            "rouge_l": rl,
            "branches_kept": result["total_branches_kept"],
            "prompt_triggers": result["prompt_triggers"],
            "time": result["timing"]["generate"],
        })

    # Summary
    print(f"\n{'='*70}")
    print("DEMO SUMMARY")
    print("=" * 70)
    print(f"  {'Case':<20} {'Jaccard':<10} {'ROUGE-L':<10} {'Branches':<10} "
          f"{'Prompts':<10} {'Time(s)':<10}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for r in results:
        print(f"  {r['name']:<20} {r['diversity']:<10} {r['rouge_l']:<10.3f} "
              f"{r['branches_kept']:<10} {r['prompt_triggers']:<10} {r['time']:<10.1f}")

    avg_div = sum(r["diversity"] for r in results) / len(results)
    avg_rl = sum(r["rouge_l"] for r in results) / len(results)
    avg_time = sum(r["time"] for r in results) / len(results)
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'AVERAGE':<20} {avg_div:<10.3f} {avg_rl:<10.3f} "
          f"{'':<10} {'':<10} {avg_time:<10.1f}")

    del model
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
