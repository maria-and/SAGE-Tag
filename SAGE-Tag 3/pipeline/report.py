"""Step 6: self-contained HTML report with an auto-drafted methods section."""
import argparse
import base64
import html
import io
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import read_all_metadata


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def draft_methods(meta: dict) -> str:
    """Template-based methods draft built only from logged parameters. Always review before use."""
    pre, sv = meta.get("preprocess", {}), meta.get("scanvi_transfer", {})
    llm, con, xa = meta.get("llm_annotate", {}), meta.get("consensus", {}), meta.get("xa_ratio", {})
    ver = (pre or sv or {}).get("versions", {})
    parts = []
    if pre:
        p = pre["params"]
        parts.append(
            f"Cells were retained if they had at least {p['min_genes']} detected genes and at most "
            f"{p['max_pct_mt']}% mitochondrial counts ({p['n_cells_input']} to {p['n_cells_after_qc']} cells), "
            f"and genes detected in fewer than {p['min_cells']} cells were removed. Counts were normalized to "
            f"10,000 per cell and log-transformed; {p['n_hvg']} highly variable genes were used for PCA "
            f"({p['n_pcs']} components), and cells were clustered with the Leiden algorithm at resolution "
            f"{p['resolution']}, giving {p['n_clusters']} clusters (scanpy {ver.get('scanpy', 'n/a')}).")
    if sv:
        p = sv["params"]
        parts.append(
            f"Cell types were annotated by semi-supervised label transfer with scANVI "
            f"(scvi-tools {ver.get('scvi-tools', 'n/a')}) from the reference '{p['reference']}' "
            f"({p['n_reference_cells']} cells, {p['n_reference_labels']} labels) using {p['n_shared_genes']} shared "
            f"genes and {p['n_hvg']} highly variable genes; cluster labels were assigned by majority vote.")
    if llm and not llm["params"].get("skipped"):
        parts.append(
            f"Independently, a large language model ({llm['params']['model']}) proposed a label for each cluster from "
            f"its top marker genes, restricted to the reference label set. Clusters where the two methods disagreed "
            f"or scANVI confidence was low were flagged for manual review"
            + (f" ({con['params']['n_needs_review']} of {con['params']['n_clusters']} clusters)." if con else "."))
    if xa:
        p = xa["params"]
        parts.append(
            f"The X:A ratio was computed per cell as the mean log-normalized expression of expressed X-linked genes "
            f"divided by that of expressed autosomal genes (cells needing at least {p['min_x_genes']} X-linked and "
            f"{p['min_autosomal_genes']} autosomal genes; {p['n_cells_with_ratio']} cells qualified) and summarized "
            f"by cell type.")
    return " ".join(parts) or "No step metadata found."


def run(args):
    import anndata as ad
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    adata = ad.read_h5ad(args.input)
    table = pd.read_csv(args.table)
    cells = pd.read_csv(args.xa_cells)
    summary = pd.read_csv(args.xa_summary)
    meta = read_all_metadata(args.meta_dir)

    # UMAP colored by consensus label
    rng = np.random.default_rng(0)
    idx = rng.choice(adata.n_obs, size=min(adata.n_obs, 100_000), replace=False)
    umap = adata.obsm["X_umap"][idx]
    labels = adata.obs["consensus_label"].astype(str).values[idx]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cats = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(cats):
        m = labels == c
        color = "lightgrey" if c == "needs_review" else cmap(i % 20)
        ax.scatter(umap[m, 0], umap[m, 1], s=2, c=[color], label=c, linewidths=0)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.legend(markerscale=5, fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    umap_b64 = _fig_to_b64(fig); plt.close(fig)

    # X:A ratio per cell type
    groups = [(k, g["xa_ratio"].dropna().values) for k, g in cells.groupby("cell_type")]
    groups = [(k, v) for k, v in groups if len(v) >= 20]
    xa_b64 = ""
    if groups:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot([v for _, v in groups], showfliers=False)  # 'labels' kwarg differs across matplotlib versions
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([k for k, _ in groups], rotation=45, ha="right", fontsize=8)
        ax.axhline(1.0, ls="--", c="grey", lw=0.8)
        ax.set_ylabel("X:A ratio")
        xa_b64 = _fig_to_b64(fig); plt.close(fig)

    shown = table[["cluster", "n_cells", "scanvi_label", "scanvi_mean_conf", "llm_label",
                   "status", "consensus_label", "top_markers"]]
    flagged = table[table["status"].isin(["discordant", "agree_low_confidence", "low_confidence"])]
    flagged_rows = "".join(
        f"<tr><td>{html.escape(str(r.cluster))}</td><td>{html.escape(r.scanvi_label)}</td>"
        f"<td>{html.escape(r.llm_label)}</td><td>{html.escape(str(r.llm_reasoning))}</td></tr>"
        for r in flagged.itertuples())

    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Annotation report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#222}}
table{{border-collapse:collapse;font-size:.85rem;width:100%}}td,th{{border:1px solid #ddd;padding:4px 8px;text-align:left}}
th{{background:#f4f4f4}}.note{{background:#fff8e1;padding:.6rem 1rem;border-left:4px solid #f5b800}}</style></head><body>
<h1>Consensus annotation report</h1>
<p>{adata.n_obs:,} cells, {len(table)} clusters; {int((table.consensus_label == 'needs_review').sum())} cluster(s) flagged for review.</p>
<h2>UMAP by consensus label</h2><img src="data:image/png;base64,{umap_b64}" style="max-width:100%">
<h2>Cluster-level comparison</h2>{shown.to_html(index=False, border=0)}
<h2>Clusters flagged for review</h2>
<table><tr><th>Cluster</th><th>scANVI</th><th>LLM</th><th>LLM reasoning</th></tr>{flagged_rows or '<tr><td colspan=4>None</td></tr>'}</table>
<h2>X:A ratio by cell type</h2>{f'<img src="data:image/png;base64,{xa_b64}" style="max-width:100%">' if xa_b64 else '<p>Not enough cells with a defined ratio.</p>'}
{summary.round(3).to_html(index=False, border=0)}
<h2>Auto-drafted methods</h2><p class="note">Generated from logged parameters. Review and edit before use.</p>
<p>{html.escape(draft_methods(meta))}</p>
</body></html>"""
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(doc)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--table", required=True)
    p.add_argument("--xa-cells", required=True)
    p.add_argument("--xa-summary", required=True)
    p.add_argument("--meta-dir", required=True)
    p.add_argument("--out", required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
