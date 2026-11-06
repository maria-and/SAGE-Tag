"""Step 4: compare scANVI and LLM cluster labels and flag disagreements for review."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .utils import write_step_metadata

NO_LLM = {"Unknown", "SKIPPED"}


def cluster_consensus(obs: pd.DataFrame, llm_labels: dict, markers: dict | None = None,
                      cluster_key: str = "cluster", scanvi_key: str = "scanvi_label",
                      conf_key: str = "scanvi_conf", min_purity: float = 0.6,
                      min_conf: float = 0.7) -> pd.DataFrame:
    """One row per cluster.

    status:
      agree                 scANVI and LLM labels match and scANVI is confident
      agree_low_confidence  labels match but scANVI purity/confidence is below threshold
      discordant            labels differ (LLM answered)
      scanvi_only           LLM unavailable/unknown; scANVI label kept if confident
      low_confidence        LLM unavailable/unknown and scANVI not confident
    consensus_label is the scANVI label for agree and confident scanvi_only clusters,
    otherwise "needs_review".
    """
    rows = []
    for cluster, grp in obs.groupby(cluster_key, observed=True):
        counts = grp[scanvi_key].value_counts()
        top = str(counts.index[0])
        purity = float(counts.iloc[0] / len(grp))
        conf = float(grp.loc[grp[scanvi_key] == top, conf_key].mean())
        confident = purity >= min_purity and conf >= min_conf

        llm = llm_labels.get(str(cluster), {})
        llm_label = str(llm.get("label", "Unknown"))
        if llm_label in NO_LLM:
            status = "scanvi_only" if confident else "low_confidence"
        elif llm_label == top:
            status = "agree" if confident else "agree_low_confidence"
        else:
            status = "discordant"

        accepted = status in {"agree", "scanvi_only"}
        rows.append({
            "cluster": str(cluster), "n_cells": int(len(grp)),
            "scanvi_label": top, "scanvi_cluster_fraction": round(purity, 3),
            "scanvi_mean_conf": round(conf, 3),
            "llm_label": llm_label, "llm_confidence": llm.get("confidence", ""),
            "llm_reasoning": llm.get("reasoning", ""),
            "status": status, "consensus_label": top if accepted else "needs_review",
            "top_markers": ", ".join((markers or {}).get(str(cluster), [])[:10]),
        })
    return pd.DataFrame(rows)


def run(args):
    import anndata as ad

    adata = ad.read_h5ad(args.input)
    llm_labels = json.loads(Path(args.llm_labels).read_text())
    markers = json.loads(Path(args.markers).read_text()) if args.markers else None
    table = cluster_consensus(adata.obs, llm_labels, markers,
                              min_purity=args.min_purity, min_conf=args.min_conf)
    mapping = dict(zip(table["cluster"], table["consensus_label"]))
    adata.obs["consensus_label"] = adata.obs["cluster"].astype(str).map(mapping).astype("category")
    table.to_csv(args.out_table, index=False)
    adata.write_h5ad(args.out_h5ad)

    write_step_metadata(args.meta_dir, "consensus", {
        "min_purity": args.min_purity, "min_conf": args.min_conf,
        "n_clusters": int(len(table)),
        "n_agree": int((table["status"] == "agree").sum()),
        "n_discordant": int((table["status"] == "discordant").sum()),
        "n_needs_review": int((table["consensus_label"] == "needs_review").sum()),
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--llm-labels", required=True)
    p.add_argument("--markers", default=None)
    p.add_argument("--out-h5ad", required=True)
    p.add_argument("--out-table", required=True)
    p.add_argument("--meta-dir", required=True)
    p.add_argument("--min-purity", type=float, default=0.6)
    p.add_argument("--min-conf", type=float, default=0.7)
    run(p.parse_args())


if __name__ == "__main__":
    main()
