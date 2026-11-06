"""Step 1: QC, normalization, Leiden clustering, and per-cluster marker genes."""
import argparse
import json
from pathlib import Path

import scanpy as sc

from .utils import write_step_metadata


def run(args):
    adata = sc.read_h5ad(args.input)
    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()  # raw counts kept for scVI/scANVI
    n_before = adata.n_obs

    adata.var["mt"] = adata.var_names.str.upper().str.startswith(args.mito_prefix.upper())
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    keep = (adata.obs["n_genes_by_counts"] >= args.min_genes) & (adata.obs["pct_counts_mt"] <= args.max_pct_mt)
    adata = adata[keep].copy()
    sc.pp.filter_genes(adata, min_cells=args.min_cells)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg, flavor="seurat")
    sc.pp.pca(adata, n_comps=args.n_pcs, mask_var="highly_variable")
    sc.pp.neighbors(adata, n_pcs=args.n_pcs)
    sc.tl.leiden(adata, resolution=args.resolution, flavor="igraph", n_iterations=2,
                 directed=False, key_added="cluster")
    sc.tl.umap(adata)
    sc.tl.rank_genes_groups(adata, "cluster", method="wilcoxon", n_genes=args.n_markers)

    names = adata.uns["rank_genes_groups"]["names"]
    markers = {str(c): [str(g) for g in names[c][: args.n_markers]] for c in names.dtype.names}

    Path(args.out_markers).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_markers).write_text(json.dumps(markers, indent=2))
    adata.write_h5ad(args.out_h5ad)

    write_step_metadata(args.meta_dir, "preprocess", {
        "n_cells_input": int(n_before), "n_cells_after_qc": int(adata.n_obs),
        "n_genes_after_filter": int(adata.n_vars), "min_genes": args.min_genes,
        "max_pct_mt": args.max_pct_mt, "min_cells": args.min_cells, "n_hvg": args.n_hvg,
        "n_pcs": args.n_pcs, "resolution": args.resolution, "n_clusters": len(markers),
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--out-h5ad", required=True)
    p.add_argument("--out-markers", required=True)
    p.add_argument("--meta-dir", required=True)
    p.add_argument("--min-genes", type=int, default=200)
    p.add_argument("--max-pct-mt", type=float, default=20.0)
    p.add_argument("--min-cells", type=int, default=3)
    p.add_argument("--mito-prefix", default="MT-", help="Case-insensitive mitochondrial gene prefix")
    p.add_argument("--n-hvg", type=int, default=3000)
    p.add_argument("--n-pcs", type=int, default=30)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--n-markers", type=int, default=25)
    run(p.parse_args())


if __name__ == "__main__":
    main()
