"""Step 3: LLM-proposed cluster annotations from marker genes (independent of scANVI).

PRIVACY: marker-gene lists are sent to an external API. Use only public or
non-sensitive data. Never run this step on unpublished or restricted datasets.
"""
import argparse
import json
import re
import time
from pathlib import Path

from .utils import write_step_metadata

PROMPT = """You are an expert single-cell biologist annotating clusters from marker genes.

Context: {context}

Cluster {cluster} top marker genes (ranked): {genes}

Choose the single best label from this allowed list, or "Unknown" if no label is well supported:
{allowed}

Respond with JSON only, no other text:
{{"label": "<one allowed label or Unknown>", "confidence": "high|medium|low", "reasoning": "<one or two sentences citing specific marker genes>"}}"""


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


def annotate_cluster(client, model, prompt, retries=3):
    for attempt in range(retries):
        try:
            msg = client.messages.create(model=model, max_tokens=400,
                                         messages=[{"role": "user", "content": prompt}])
            return parse_json(msg.content[0].text)
        except Exception as exc:  # noqa: BLE001 - retry on API or parse errors
            if attempt == retries - 1:
                return {"label": "Unknown", "confidence": "low", "reasoning": f"LLM call failed: {exc}"}
            time.sleep(2 ** attempt)


def run(args):
    markers = json.loads(Path(args.markers).read_text())
    allowed = json.loads(Path(args.allowed_labels).read_text())
    results = {}

    if args.skip_llm:
        results = {c: {"label": "SKIPPED", "confidence": "n/a", "reasoning": "LLM step skipped"} for c in markers}
    else:
        import anthropic  # reads ANTHROPIC_API_KEY from the environment

        client = anthropic.Anthropic()
        for cluster, genes in markers.items():
            prompt = PROMPT.format(context=args.context, cluster=cluster,
                                   genes=", ".join(genes[: args.n_genes]),
                                   allowed="; ".join(allowed))
            res = annotate_cluster(client, args.model, prompt)
            if res.get("label") not in allowed:  # enforce the allowed vocabulary
                res["label"] = "Unknown"
            results[cluster] = res

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    write_step_metadata(args.meta_dir, "llm_annotate", {
        "model": args.model, "skipped": bool(args.skip_llm), "n_marker_genes_per_cluster": args.n_genes,
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--markers", required=True)
    p.add_argument("--allowed-labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--meta-dir", required=True)
    p.add_argument("--context", default="single-cell RNA-seq of tissue (specify tissue and species)")
    p.add_argument("--model", default="claude-sonnet-5-5")
    p.add_argument("--n-genes", type=int, default=20)
    p.add_argument("--skip-llm", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
