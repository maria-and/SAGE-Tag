"""Step 2: scANVI label transfer from an annotated reference to the query data."""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from .utils import write_step_metadata


def _counts_matrix(adata, genes):
    sub = adata[:, genes]
    return (sub.layers["counts"] if "counts" in sub.layers else sub.X).copy()


def run(args):
    import scvi  # imported lazily: heavy dependency

    query = sc.read_h5ad(args.query)
    ref = sc.read_h5ad(args.reference)
    if args.label_key not in ref.obs:
        raise KeyError(f"Reference has no obs column '{args.label_key}'")
    ref.var_names_make_unique()

    shared = query.var_names.intersection(ref.var_names)
    if len(shared) < 500:
        raise ValueError(f"Only {len(shared)} shared genes between query and reference; check gene identifiers.")

    ref_a = ad.AnnData(X=_counts_matrix(ref, shared),
                       obs=pd.DataFrame(index="ref_" + ref.obs_names.astype(str)),
                       var=pd.DataFrame(index=shared))
    ref_a.obs["celltype"] = ref.obs[args.label_key].astype(str).values
    ref_a.obs["batch"] = "reference"
    qry_a = ad.AnnData(X=_counts_matrix(query, shared),
                       obs=pd.DataFrame(index="query_" + query.obs_names.astype(str)),
                       var=pd.DataFrame(index=shared))
    qry_a.obs["celltype"] = "Unknown"
    qry_a.obs["batch"] = "query"

    n_query = qry_a.n_obs
    combined = ad.concat([ref_a, qry_a], join="inner")  # query rows are last
    combined.layers["counts"] = combined.X.copy()
    sc.pp.highly_variable_genes(combined, n_top_genes=args.n_hvg, flavor="seurat_v3",
                                batch_key="batch", layer="counts", subset=True)

    scvi.model.SCVI.setup_anndata(combined, layer="counts", batch_key="batch",
                                  labels_key="celltype", unlabeled_category="Unknown")
    vae = scvi.model.SCVI(combined, n_latent=30, n_layers=2)
    vae.train(max_epochs=args.scvi_epochs)
    lvae = scvi.model.SCANVI.from_scvi_model(vae, unlabeled_category="Unknown", labels_key="celltype")
    lvae.train(max_epochs=args.scanvi_epochs, n_samples_per_label=100)

    probs = lvae.predict(soft=True).iloc[-n_query:]
    query.obs["scanvi_label"] = probs.idxmax(axis=1).values
    query.obs["scanvi_conf"] = probs.max(axis=1).values.astype(float)
    query.obsm["X_scANVI"] = np.asarray(lvae.get_latent_representation())[-n_query:]

    allowed = sorted(set(ref_a.obs["celltype"]) - {"Unknown"})
    Path(args.out_labels).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_labels).write_text(json.dumps(allowed, indent=2))
    query.write_h5ad(args.out_h5ad)

    write_step_metadata(args.meta_dir, "scanvi_transfer", {
        "reference": Path(args.reference).name, "label_key": args.label_key,
        "n_reference_cells": int(ref.n_obs), "n_reference_labels": len(allowed),
        "n_shared_genes": int(len(shared)), "n_hvg": args.n_hvg,
        "scvi_epochs": args.scvi_epochs, "scanvi_epochs": args.scanvi_epochs,
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--query", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--label-key", required=True)
    p.add_argument("--out-h5ad", required=True)
    p.add_argument("--out-labels", required=True)
    p.add_argument("--meta-dir", required=True)
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--scvi-epochs", type=int, default=100)
    p.add_argument("--scanvi-epochs", type=int, default=20)
    run(p.parse_args())


if __name__ == "__main__":
    main()
