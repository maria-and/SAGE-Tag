# pero-xist

An automated Snakemake pipeline for single-nucleus RNA data that **cross-checks machine-learning and LLM cell-type annotations** and flags the clusters where they disagree, so a human reviews only the uncertain calls.

```
raw counts ──► QC + Leiden clusters + markers
                    │
                    ├──► scANVI label transfer from an annotated reference   (ML)
                    └──► LLM label proposal from cluster marker genes         (LLM)
                                   │
                          consensus: agree / discordant / low-confidence
                                   │
                    per-cell-type X:A ratio ──► HTML report + auto-drafted methods
```

## What it does

| Step | Module | Output |
|---|---|---|
| 1. QC, normalize, cluster, markers | `pipeline/preprocess.py` | clustered `.h5ad`, `markers.json` |
| 2. scANVI label transfer | `pipeline/scanvi_transfer.py` | per-cell labels and confidence |
| 3. LLM annotation (optional) | `pipeline/llm_annotate.py` | per-cluster label, confidence, reasoning |
| 4. Consensus and review flags | `pipeline/consensus.py` | `consensus_table.csv`, `consensus_label` |
| 5. X:A ratio per cell type | `pipeline/xa_ratio.py` | per-cell and per-cell-type ratios |
| 6. Report | `pipeline/report.py` | self-contained `report.html` with a methods draft |

Cluster status values: `agree`, `agree_low_confidence`, `discordant`, `scanvi_only` (LLM unavailable), `low_confidence`. Only `agree` and confident `scanvi_only` clusters receive a consensus label; everything else is `needs_review`.

## Quickstart

```bash
pip install -r requirements.txt
cp your_query.h5ad data/query.h5ad          # raw counts in .X
cp your_reference.h5ad data/reference.h5ad  # annotated reference
# edit config/config.yaml (reference_label_key, llm.context, ...)
export ANTHROPIC_API_KEY=...                # only if llm.skip is false
snakemake --cores 8
```

Docker: `docker build -t pero-xist . && docker run -v $PWD:/app pero-xist`

Run without any external API call by setting `llm.skip: true` in the config.

## Data policy

- **Use public data for demos.** Do not commit data, results, or figures from unpublished or restricted datasets.
- The LLM step sends **marker-gene lists** to an external API. Run it only on data you are permitted to share that way.

## Notes and limitations

- LLM labels can be confidently wrong. They are used as an independent check on scANVI, not as ground truth, and are restricted to the reference label vocabulary.
- X:A is a simple estimator (ratio of mean log-normalized expression among expressed X-linked and autosomal genes). It needs a `chromosome` column in `adata.var`, or a TSV (`gene`, `chromosome`) via `gene_annotation` in the config. Validate it on your system before interpreting it.
- Doublet removal is not included; run it upstream if needed.
- The methods paragraph is generated from logged parameters and package versions. Review it before using it in a manuscript.
- Unit tests cover the consensus logic and X:A estimator (`pytest`). End-to-end runs of the scANVI and LLM steps require `scvi-tools` and an API key.

## License

MIT
