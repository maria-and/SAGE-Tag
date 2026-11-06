# SAGE-Tag

**Species-Aware Genomic and Epigenomic analysis of Paired-Tag** — a research-stage toolkit for exploring X-chromosome regulation across *Mus musculus* and *Peromyscus leucopus*.

**Status: research prototype.** The bulk importer and integration interfaces are implemented and covered by synthetic-data tests. No results have been validated against the user's real data. UCE inference, literature curation, and full automated Seurat-to-model execution are **not** included. Model scores are exploratory rankings, **not** verified XCI escape calls or evidence of Xist causality.

## Study design

| Source | Tissue | Animals | Data | Interpretation |
| --- | --- | --- | --- | --- |
| Paired-Tag | Liver | Separate animals | Single-nucleus RNA + H3K27me3 | Cell-type/Xist-status descriptive contrasts |
| Bulk RNA-seq | Dermal fibroblasts | Separate animals | TPM (2 male, 3 female per species in provided tables) | Expression and sex contrasts |
| Bulk CUT&Tag | Dermal fibroblasts | Separate animals | H3K27me3 gene RPKM (2 male, 2 female per species) | Chromatin-enrichment proxy and sex contrasts |

**Never treat liver and fibroblasts as paired samples.** Tissue, assay, and animal differences are confounded. Cross-tissue agreement is supporting evidence, not a controlled tissue effect.

## Install

Python 3.10+ recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python sage.py --help
```

## 1. Import your bulk fibroblast CSVs

```bash
python sage.py bulk \
  --mus-h3k "/Volumes/Extreme Pro/CutandTag_XiXa/mus_rpkm_with_chromosome.csv" \
  --pero-h3k "/Volumes/Extreme Pro/CutandTag_XiXa/pero_rpkm_with_chromosome.csv" \
  --mus-tpm "$HOME/Desktop/Docs/musgenome-tpm.csv" \
  --pero-tpm "$HOME/Desktop/Docs/perogenome-tpm.csv" \
  --out-dir outputs
```

This writes `fibro_mus_female.csv`, `fibro_pero_female.csv`, `fibro_mus_male.csv`, `fibro_pero_male.csv`, and an import summary. RNA and CUT&Tag are joined **within species only**. Chromosome annotations must be present in each species' H3K table for X-only filtering. If absent, use `--all-genes` and annotate/filter X genes later. Verify chromosome assembly and gene identifiers. **The input values are processed TPM/RPKM, not raw reads**; the importer does not perform allele-specific analysis or valid count-based differential-expression testing.

Female sex contrasts use all three RNA female replicates, and two CUT&Tag female replicates. Male fibroblast `rna_log2fc`/`h3k_log2fc` mean **male versus female within Peromyscus**; they are *not* the male Peromyscus-versus-Mus contrast from the original R analysis and are *not* Xist-positive versus Xist-negative contrasts. `male_repression_signature` is descriptive only.

## 2. Prepare liver features

The existing R export and feature-building modules are available for development:

```bash
Rscript R/export_pairedtag.R "/path/to/ShaSun_H3K27me3_mm.rawSeurat.RDS" Mus data
Rscript R/export_pairedtag.R "/path/to/ShaSun_H3K27me3_pl.rawSeurat.RDS" Pero data
python -m pipeline.build_features --input-dir data --species Mus --annotation config/mus_genes.tsv --out-features outputs/mus_features.csv --out-contrasts outputs/mus_contrasts.csv
python -m pipeline.build_features --input-dir data --species Pero --annotation config/pero_genes.tsv --out-features outputs/pero_features.csv --out-contrasts outputs/pero_contrasts.csv
```

These steps need species-specific gene annotations and inspection of Seurat assay names, sex/sample assignments, Xist IDs, reference contigs, and library-size normalization. Build **female one-gene-per-row** liver tables with the same feature definitions for Mus and Peromyscus before integration. See `pipeline/aggregate_features.py` for an exploratory aggregator. Nuclei are not independent biological replicates; use sample-aware methods for inference.

## 3. Curate orthologs and literature labels

Populate (do not leave header-only):

- `config/orthologs.csv`: `mus_gene,pero_gene` (reviewed **one-to-one** orthologs, matching feature-table IDs)
- `config/mus_xci_labels.csv`: `mus_gene,xci_status,source_doi` (`subject` or `escape`, based on actual published female Xi evidence)

Template headers are in `config/*.template.csv`. No labels or orthologs are fabricated or bundled. Relevant starting points: Yang et al. 2010 (DOI 10.1101/gr.103200.109); Berletch et al. 2015 (DOI 10.1371/journal.pgen.1005079). Validate labels against the original tables and tissue context.

## 4. Cross-tissue integration

Run fibroblast-only initially (requires real labels and orthologs):

```bash
python sage.py integrate \
  --fibro-mus-female outputs/fibro_mus_female.csv \
  --fibro-pero-female outputs/fibro_pero_female.csv \
  --fibro-pero-male outputs/fibro_pero_male.csv \
  --orthologs config/orthologs.csv \
  --labels config/mus_xci_labels.csv \
  --out outputs/xci_candidates.csv \
  --metrics outputs/xci_metrics.json
```

When liver feature tables have been validated, append `--liver-mus-female`, `--liver-pero-female`, and optionally `--liver-pero-male`. **Each tissue has its own classifier**; predictions are compared, not fitted as matched observations. Cross-validation on Mus genes does not validate transfer to Peromyscus. Avoid interpreting tissue-average model scores as calibrated escape probabilities.

## 5. Optional UCE foundation-model integration

UCE is an **external** single-cell RNA foundation model. This repo **does not install model weights or run UCE inference**. Run the official model on suitable liver RNA counts with supported species/gene identifiers, then attach exported `cell_id,uce_0,uce_1,...` embeddings:

```bash
python sage.py uce-attach --metadata data/liver_metadata.csv --embeddings data/uce_embeddings.csv --out outputs/liver_with_uce.csv
```

This is an embedding join and validation step, not UCE-based XCI prediction. Check cross-species gene support and batch effects before interpreting alignment.

## Optional allele-specific feasibility

`pipeline/allele_specific.py` is an exploratory allele-counting module requiring **original alignment BAM/CRAM, per-animal heterozygous SNP VCF, and appropriate gene intervals**; processed TPM/RPKM and Seurat RDS alone are insufficient. For outbred animals, call/validate SNPs **per animal**. Female random XCI makes pooled biallelic expression insufficient evidence for escape.

## Repository layout

- `sage.py` — user-facing CLI for bulk, integration, and UCE attachment
- `pipeline/import_bulk_fibroblast.py` — CSV input and feature engineering
- `pipeline/cross_tissue_xci.py` — tissue-specific supervised models and ranking
- `pipeline/predict_female_escape.py` — female escape-like classifier
- `pipeline/uce_bridge.py` — external UCE embedding attachment
- `pipeline/allele_specific.py` — exploratory allelic counting
- `R/export_pairedtag.R` — Seurat export helper
- `tests/` — synthetic-data tests

## Scientific limitations

1. Escape-like scores **do not prove expression from the inactive X**; Xi-aware allelic evidence is required.
2. Male RNA-down/H3K27me3-up signatures **do not establish direct Xist-mediated repression**.
3. H3K27me3 RPKM is not equivalent to calibrated occupancy; check coverage, controls, and library effects.
4. With two bulk biological replicates for several groups, treat gene-wise p-values cautiously; prefer count-based, replicate-aware methods when raw counts are available.
5. Do not upload patient/private, unpublished sequencing data, credentials, or large raw files to a public GitHub repository.

## License

No license has been selected. Add a `LICENSE` file before inviting external reuse.
