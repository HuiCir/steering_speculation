"""
Steering-Speculation Benchmark Script
======================================
Runs all methods (Batch, SPREAD, Spread+Prompt, COMBINED)
on all sample cases and produces a comparison table.
"""
import sys
import time
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    Orchestrator,
    load_sample,
    discover_samples,
    jaccard_diversity,
    rouge_l_score,
    make_fixed_hook,
    make_gated_hook,
    select_phase,
    select_diverse_branches,
)


def generate_batch(model, tokenizer, prompt, N=4, max_tok=200):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    ilen = inp["input_ids"].shape[1]
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tok, do_sample=True,
                             temperature=0.7, top_p=0.9, num_return_sequences=N,
                             pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize(); et = time.time() - t0
    res = [tokenizer.decode(out[i][ilen:], skip_special_tokens=True) for i in range(out.shape[0])]
    del out, inp; torch.cuda.empty_cache()
    return res, et


def generate_spread(model, tokenizer, prompt, N=4, max_tok=200):
    hooks = make_fixed_hook(model, 8, 0.3, 20)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    ilen = inp["input_ids"].shape[1]
    torch.cuda.synchronize(); t0 = time.time()
    try:
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_tok, do_sample=True,
                                 temperature=0.7, top_p=0.9, num_return_sequences=N,
                                 pad_token_id=tokenizer.eos_token_id)
    finally:
        for h in hooks: h.remove()
    torch.cuda.synchronize(); et = time.time() - t0
    res = [tokenizer.decode(out[i][ilen:], skip_special_tokens=True) for i in range(out.shape[0])]
    del out, inp; torch.cuda.empty_cache()
    return res, et


def main():
    parser = argparse.ArgumentParser(description="Steering-Speculation Benchmark")
    parser.add_argument("--model", required=True,
                        help="Path to HuggingFace model (e.g., /path/to/qwen3.5-2b)")
    parser.add_argument("--samples", default="samples")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--max-cases", type=int, default=10)
    args = parser.parse_args()

    print("=" * 70)
    print("Steering-Speculation Benchmark")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    print(f"[Load] {sum(p.numel() for p in model.parameters())/1e9:.2f}B, {time.time()-t0:.1f}s")

    orch = Orchestrator(model, tokenizer)
    samples = discover_samples(args.samples)[: args.max_cases]

    all_results = []
    for name, filepath in samples:
        case = load_sample(filepath)
        print(f"\n{'─'*50}\n[{name}]")
        cr = {"name": name, "methods": {}}

        # Batch
        r, t = generate_batch(model, tokenizer, case["prompt"])
        cr["methods"]["Batch"] = {"jaccard": jaccard_diversity(r), "time": t,
                                   "rouge_l": max(rouge_l_score(rr, case["reference"]) for rr in r)}

        # SPREAD
        r, t = generate_spread(model, tokenizer, case["prompt"])
        cr["methods"]["SPREAD"] = {"jaccard": jaccard_diversity(r), "time": t,
                                    "rouge_l": max(rouge_l_score(rr, case["reference"]) for rr in r)}

        # Spread+Prompt (phase1 only)
        result = orch.run(case["prompt"])
        r = result["responses"]
        cr["methods"]["Sprd+Prompt"] = {"jaccard": jaccard_diversity(r),
                                         "time": result["timing"]["generate"],
                                         "rouge_l": max(rouge_l_score(rr, case["reference"]) for rr in r),
                                         "prompts": result["prompt_triggers"]}

        # COMBINED
        cr["methods"]["COMBINED"] = {"jaccard": result["diversity"],
                                      "time": result["timing"]["generate"],
                                      "rouge_l": max(rouge_l_score(rr, case["reference"]) for rr in result["responses"]),
                                      "branches_kept": result["total_branches_kept"]}

        for mn, m in cr["methods"].items():
            print(f"  {mn:<15} J={m['jaccard']:.3f} R-L={m.get('rouge_l',0):.3f} t={m['time']:.1f}s")

        all_results.append(cr)

    # Aggregate
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for label, methods in [("ALL", all_results),
                            ("Parallel", [r for r in all_results if r["name"].startswith("P")]),
                            ("Sequential", [r for r in all_results if r["name"].startswith("S")])]:
        if not methods: continue
        print(f"\n  {label} ({len(methods)} cases):")
        print(f"  {'Method':<15} {'Jaccard':<10} {'ROUGE-L':<10} {'Time(s)':<10}")
        print(f"  {'─'*15} {'─'*10} {'─'*10} {'─'*10}")
        for mn in ["Batch", "SPREAD", "Sprd+Prompt", "COMBINED"]:
            js = [r["methods"][mn]["jaccard"] for r in methods if mn in r["methods"]]
            rs = [r["methods"][mn]["rouge_l"] for r in methods if mn in r["methods"]]
            ts = [r["methods"][mn]["time"] for r in methods if mn in r["methods"]]
            if js:
                print(f"  {mn:<15} {np.mean(js):<10.3f} {np.mean(rs):<10.3f} {np.mean(ts):<10.1f}")

    # Save
    Path(args.output).write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\n[Save] {args.output}")

    del model; torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
