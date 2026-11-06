"""Step 5: per-cell and per-cell-type X:A expression ratio.

Estimator: for each cell, mean log-normalized expression over expressed X-linked
genes divided by the same quantity over expressed autosomal genes. Cells with too
few expressed genes in either set get NaN. This is a simple, transparent estimator;
validate it against your own data before drawing biological conclusions.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .utils import write_step_metadata

NON_AUTOSOMES = {"X", "Y", "MT", "M"}


def _norm(chrom) -> pd.Series:
    return pd.Series(chrom).astype(str).str.upper().str.replace(r"^CHR", "", regex=True)


def _mean_over_expressed(X, mask):
    """Mean of non-zero entries per cell over the masked genes, plus n expressed genes."""
    idx = np.flatnonzero(mask)
    mean = np.full(X.shape[0], np.nan)
    if idx.size == 0:
        return mean, np.zeros(X.shape[0], dtype=int)
    sub = sp.csr_matrix(X[:, idx])
    sub.eliminate_zeros()
    sums = np.asarray(sub.sum(axis=1)).ravel()
    nnz = np.diff(sub.indptr)
    ok = nnz > 0
    mean[ok] = sums[ok] / nnz[ok]
    return mean, nnz


def xa_ratio_per_cell(X, chromosomes, x_name="X", autosomes=None,
                      min_x_genes=10, min_autosomal_genes=100) -> np.ndarray:
    chrom = _norm(chromosomes).values
    x = _norm([x_name]).iloc[0]
    is_x = chrom == x
    if autosomes is None:
        is_a = ~np.isin(chrom, list(NON_AUTOSOMES | {x, "NAN", ""}))
    else:
        is_a = np.isin(chrom, _norm(autosomes).values)
    mx, nx = _mean_over_expressed(X, is_x)
    ma, na = _mean_over_expressed(X, is_a)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = mx / ma
    ratio[(nx < min_x_genes) | (na < min_autosomal_genes)] = np.nan
    return ratio


def summarize_by_group(ratios, groups) -> pd.DataFrame:
    df = pd.DataFrame({"cell_type": np.asarray(groups).astype(str), "xa": ratios}).dropna()
    return (df.groupby("cell_type").xa
              .agg(n_cells="size", median_xa="median",
                   q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
              .reset_index())


def run(args):
    import anndata as ad

    adata = ad.read_h5ad(args.input)
    if args.gene_annotation:
        ann = pd.read_csv(args.gene_annotation, sep="\t").set_index("gene")
        adata.var[args.chrom_key] = ann["chromosome"].reindex(adata.var_names).values
    if args.chrom_key not in adata.var:
        raise KeyError(f"adata.var has no '{args.chrom_key}' column; pass --gene-annotation (TSV: gene, chromosome)")

    autosomes = args.autosomes.split(",") if args.autosomes else None
    ratios = xa_ratio_per_cell(adata.X, adata.var[args.chrom_key].values, args.x_name, autosomes,
                               args.min_x_genes, args.min_autosomal_genes)
    label_key = args.label_key if args.label_key in adata.obs else "cluster"
    per_cell = pd.DataFrame({"cell": adata.obs_names, "cell_type": adata.obs[label_key].astype(str).values,
                             "xa_ratio": ratios})
    Path(args.out_cells).parent.mkdir(parents=True, exist_ok=True)
    per_cell.to_csv(args.out_cells, index=False)
    summarize_by_group(ratios, per_cell["cell_type"]).to_csv(args.out_summary, index=False)

    write_step_metadata(args.meta_dir, "xa_ratio", {
        "x_name": args.x_name, "min_x_genes": args.min_x_genes,
        "min_autosomal_genes": args.min_autosomal_genes,
        "n_cells_with_ratio": int(np.isfinite(ratios).sum()), "label_key": label_key,
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--out-cells", required=True)
    p.add_argument("--out-summary", required=True)
    p.add_argument("--meta-dir", required=True)
    p.add_argument("--gene-annotation", default=None, help="TSV with columns: gene, chromosome")
    p.add_argument("--chrom-key", default="chromosome")
    p.add_argument("--x-name", default="X")
    p.add_argument("--autosomes", default=None, help="Comma-separated autosome names (default: all non-X/Y/MT)")
    p.add_argument("--label-key", default="consensus_label")
    p.add_argument("--min-x-genes", type=int, default=10)
    p.add_argument("--min-autosomal-genes", type=int, default=100)
    run(p.parse_args())


if __name__ == "__main__":
    main()
