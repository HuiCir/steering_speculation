"""
Data loading utilities.
"""
import json
from pathlib import Path


def load_sample(filepath):
    """
    Load a single sample case from JSON.

    Expected format:
    {
        "query": "...",
        "tool_list": [{"tool name": "...", "executed_output": "...", ...}, ...],
        "reference": "..."  (optional ground-truth answer)
    }

    Returns:
        dict with keys: query, prompt, reference, tool_count
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both single object and list
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError("Empty sample file")
        entry = data[0]
    else:
        entry = data

    query = entry.get("query", "")
    tool_list = entry.get("tool_list", entry.get("tool list", []))
    reference = entry.get("reference", entry.get("final_answer", ""))

    if isinstance(reference, dict):
        reference = reference.get("answer", str(reference))
    reference = str(reference)[:1000]

    # Build prompt from query + tool results
    tools_text = "\n".join(
        f"  {t.get('tool name', t.get('tool_name', '?'))}: "
        f"{str(t.get('executed_output', t.get('execution_status', '?')))[:200]}..."
        for t in tool_list
    )

    prompt = (
        f"Analyze the tool results and answer the query.\n\n"
        f"Query: {query}\n\n"
        f"Tool Results:\n{tools_text}\n\n"
        f"Analysis:"
    )

    return {
        "query": query[:300],
        "prompt": prompt,
        "reference": reference,
        "tool_count": len(tool_list),
    }


def discover_samples(samples_dir):
    """
    Discover all sample JSON files in a directory tree.

    Returns:
        list of (name, filepath) tuples
    """
    samples = []
    base = Path(samples_dir)
    for fp in sorted(base.rglob("*.json")):
        rel = fp.relative_to(base)
        name = str(rel.with_suffix("")).replace("/", "-").replace("\\", "-")
        samples.append((name, str(fp)))
    return samples
