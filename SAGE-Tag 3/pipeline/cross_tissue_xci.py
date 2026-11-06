"""Tissue-aware cross-species XCI candidate prioritization.

Each tissue's Mus->Pero female escape classifier is trained independently.
Fibroblast bulk and liver Paired-Tag are NEVER treated as matched samples.
Male evidence is a descriptive RNA-down/H3K27me3-up signature, not a trained
male-Xist causality model. No allele-specific escape is inferred.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.predict_female_escape import predict


def _load(path, required):
    df = pd.read_csv(path)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing {sorted(missing)}')
    return df


def _unique(df, column, context):
    if df[column].isna().any() or df[column].duplicated().any():
        raise ValueError(f'{context}: {column} must be unique and nonmissing')


def _male_evidence(df, prefix):
    """Expected precomputed sample-aware log2 contrasts (RNA, H3K27me3).

    Caller must ensure contrasts are biologically comparable. In liver these
    may be Xist+ vs Xist-; fibroblasts may use another explicit contrast.
    """
    required = ['gene', 'rna_log2fc', 'h3k_log2fc']
    if df is None:
        return None
    if not set(required).issubset(df):
        raise ValueError(f'{prefix} male features require {required}')
    _unique(df, 'gene', prefix)
    out = df[required].copy()
    for c in required[1:]:
        out[c] = pd.to_numeric(out[c], errors='raise')
    out[f'{prefix}_male_repression_signature'] = (
        (out.rna_log2fc < 0) & (out.h3k_log2fc > 0)
    ).where(out[['rna_log2fc', 'h3k_log2fc']].notna().all(axis=1))
    return out.rename(columns={'gene': 'pero_gene',
                               'rna_log2fc': f'{prefix}_male_rna_log2fc',
                               'h3k_log2fc': f'{prefix}_male_h3k_log2fc'})


def integrate(liver, fibro, orthologs, labels, liver_male=None, fibro_male=None,
              liver_features=('rna_mean', 'h3k_mean', 'rna_log2fc', 'h3k_log2fc'),
              fibro_features=('rna_mean', 'h3k_mean', 'rna_log2fc', 'h3k_log2fc')):
    """liver/fibro: (Mus female table, Pero female table).

    Each pair must use the SAME feature definitions within that tissue. The
    two tissue feature spaces may differ, because models are fitted separately.
    """
    _unique(orthologs, 'mus_gene', 'orthologs')
    _unique(orthologs, 'pero_gene', 'orthologs')
    out = orthologs[['mus_gene', 'pero_gene']].copy()
    metrics = {}
    for tissue, pair, features in [('liver', liver, liver_features),
                                   ('fibroblast', fibro, fibro_features)]:
        if pair is None:
            metrics[tissue] = {'status': 'not_provided'}
            continue
        preds, stats, _ = predict(pair[0], pair[1], orthologs, labels,
                                  features=features)
        _unique(preds, 'mus_gene', f'{tissue} predictions')
        keep = preds[['mus_gene', 'escape_like_probability']].rename(
            columns={'escape_like_probability': f'{tissue}_escape_like_score'})
        out = out.merge(keep, on='mus_gene', how='left', validate='one_to_one')
        metrics[tissue] = stats
    if 'liver_escape_like_score' not in out and 'fibroblast_escape_like_score' not in out:
        raise ValueError('At least one female tissue must be supplied')
    for tissue, male in [('liver', liver_male), ('fibroblast', fibro_male)]:
        evidence = _male_evidence(male, tissue)
        if evidence is not None:
            out = out.merge(evidence, on='pero_gene', how='left', validate='one_to_one')
    labelcols = labels[['mus_gene', 'xci_status', 'source_doi']].rename(
        columns={'xci_status': 'mus_literature_xci_status',
                 'source_doi': 'mus_literature_source_doi'})
    out = out.merge(labelcols, on='mus_gene', how='left', validate='one_to_one')
    scorecols = [c for c in ['liver_escape_like_score', 'fibroblast_escape_like_score'] if c in out]
    # Rank is explicitly descriptive: average of *separately fitted* model scores.
    # Do not interpret as a calibrated escape probability.
    out['mean_escape_like_score'] = out[scorecols].mean(axis=1)
    out['n_tissues_scored'] = out[scorecols].notna().sum(axis=1)
    if len(scorecols) == 2:
        a, b = (out[c] for c in scorecols)
        out['escape_score_difference_liver_minus_fibroblast'] = a - b
        out['escape_evidence_class'] = np.select(
            [a.isna() | b.isna(), (a >= .8) & (b >= .8),
             (a <= .2) & (b <= .2), ((a >= .8) & (b <= .2)) | ((b >= .8) & (a <= .2))],
            ['missing_tissue', 'high_both', 'low_both', 'discordant_extremes'],
            default='mixed_or_uncertain')
    else:
        out['escape_evidence_class'] = 'single_tissue_only'
    for c in ['liver_male_repression_signature', 'fibroblast_male_repression_signature']:
        if c not in out:
            out[c] = pd.Series(pd.NA, index=out.index, dtype='boolean')
    a = out.liver_male_repression_signature.astype('boolean')
    b = out.fibroblast_male_repression_signature.astype('boolean')
    out['male_signature_class'] = np.select(
        [(a == True).fillna(False) & (b == True).fillna(False),
         (a == True).fillna(False) & (b == False).fillna(False),
         (a == False).fillna(False) & (b == True).fillna(False),
         (a == False).fillna(False) & (b == False).fillna(False)],
        ['both', 'liver_only', 'fibroblast_only', 'neither'],
        default='missing_or_uncertain')
    out = out.sort_values('mean_escape_like_score', ascending=False, na_position='last')
    metrics['limitations'] = [
        'Bulk fibroblasts and liver Paired-Tag are from different animals and tissues.',
        'Cross-tissue scores are separate model rankings, not calibrated probabilities.',
        'Female XCI escape requires allele-specific Xi expression validation.',
        'Male repression signatures are descriptive and may reflect different contrasts.',
        'Mus gene-wise cross-validation does not validate Peromyscus transfer.',
        'Do not interpret tissue differences as causal tissue effects; assay and animal differ.',
    ]
    return out.reset_index(drop=True), metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ['orthologs', 'labels', 'out', 'metrics']:
        p.add_argument('--' + flag, required=True)
    for flag in ['liver-mus-female', 'liver-pero-female',
                 'fibro-mus-female', 'fibro-pero-female',
                 'liver-pero-male', 'fibro-pero-male']:
        p.add_argument('--' + flag)
    p.add_argument('--liver-features', nargs='+', default=['rna_mean', 'h3k_mean', 'rna_log2fc', 'h3k_log2fc'])
    p.add_argument('--fibro-features', nargs='+', default=['rna_mean', 'h3k_mean', 'rna_log2fc', 'h3k_log2fc'])
    args = p.parse_args()
    orth = _load(args.orthologs, ['mus_gene', 'pero_gene'])
    labels = _load(args.labels, ['mus_gene', 'xci_status', 'source_doi'])
    def pair(mus, pero, tissue):
        if bool(mus) != bool(pero):
            raise ValueError(f'{tissue}: provide BOTH Mus and Pero female tables')
        return (_load(mus, ['gene']), _load(pero, ['gene'])) if mus else None
    liver = pair(args.liver_mus_female, args.liver_pero_female, 'liver')
    fibro = pair(args.fibro_mus_female, args.fibro_pero_female, 'fibroblast')
    def male(path):
        return _load(path, ['gene', 'rna_log2fc', 'h3k_log2fc']) if path else None
    out, metrics = integrate(liver, fibro, orth, labels,
                             male(args.liver_pero_male), male(args.fibro_pero_male),
                             args.liver_features, args.fibro_features)
    for dest in [args.out, args.metrics]:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2))
    print(f'Wrote {len(out)} orthologs to {args.out}')

if __name__ == '__main__':
    main()
