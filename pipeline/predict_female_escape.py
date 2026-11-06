"""Literature-supervised female XCI escape *candidate* scoring across species.

IMPORTANT: Non-allele-specific Paired-Tag cannot establish escape from the inactive X.
This model estimates resemblance to curated mouse escape signatures, not Xi expression.
Train only on matched female feature definitions, ideally Xi-allele-specific references.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Shared feature definitions MUST represent comparable assays, tissues and female cell types.
DEFAULT_FEATURES = ('rna_mean', 'h3k_mean', 'rna_log2fc', 'h3k_log2fc')


def _validate_unique(df, column, name):
    if df[column].isna().any() or df[column].duplicated().any():
        raise ValueError(f'{name}: {column} must be nonmissing and unique')


def _require(df, cols, name):
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f'{name} missing columns: {sorted(missing)}')


def _validate_features(df, features, name):
    _require(df, ['gene', *features], name)
    _validate_unique(df, 'gene', name)
    if not all(pd.api.types.is_numeric_dtype(df[f]) for f in features):
        raise ValueError(f'{name}: feature columns must be numeric')
    if np.isinf(df[list(features)].to_numpy(dtype=float)).any():
        raise ValueError(f'{name}: features contain infinity')
    if df[list(features)].isna().all(axis=0).any():
        raise ValueError(f'{name}: at least one feature is entirely missing')


def predict(mus_female, pero_female, orthologs, labels, features=DEFAULT_FEATURES,
            min_class=5, n_splits=5, seed=42):
    """Return Peromyscus escape-like probabilities, CV metrics and held-out Mus predictions.

    Labels: mus_gene, xci_status ('subject'/'escape'), source_doi; optional tissue.
    A high score is a candidate prioritization signal, not evidence of Xi transcription.
    """
    features = tuple(features)
    if not features or len(set(features)) != len(features):
        raise ValueError('Provide a nonempty, unique list of features')
    _validate_features(mus_female, features, 'Mus female features')
    _validate_features(pero_female, features, 'Peromyscus female features')
    _require(orthologs, ['mus_gene', 'pero_gene'], 'orthologs')
    _validate_unique(orthologs, 'mus_gene', 'orthologs')
    _validate_unique(orthologs, 'pero_gene', 'orthologs')
    _require(labels, ['mus_gene', 'xci_status', 'source_doi'], 'literature labels')
    _validate_unique(labels, 'mus_gene', 'literature labels')
    if not labels.xci_status.isin(['subject', 'escape']).all():
        raise ValueError('xci_status must be subject or escape')
    if labels.source_doi.isna().any() or (labels.source_doi.astype(str).str.strip() == '').any():
        raise ValueError('Every label must have a literature DOI')

    train = mus_female.merge(labels, left_on='gene', right_on='mus_gene', validate='one_to_one')
    if train.empty or train.xci_status.value_counts().reindex(['escape', 'subject'], fill_value=0).min() < min_class:
        raise ValueError(f'Require at least {min_class} labeled genes in each class after matching features')
    if train[list(features)].isna().all(axis=0).any():
        raise ValueError('Training features have an entirely missing column')
    target = pero_female.merge(orthologs, left_on='gene', right_on='pero_gene', validate='one_to_one')
    if target.empty:
        raise ValueError('No Peromyscus genes matched the ortholog table')

    y = (train.xci_status == 'escape').astype(int)
    model = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=3000, random_state=seed)),
    ])
    cv = StratifiedKFold(n_splits=min(n_splits, int(y.value_counts().min())), shuffle=True, random_state=seed)
    heldout = cross_val_predict(model, train[list(features)], y, cv=cv, method='predict_proba')[:, 1]
    metrics = {
        'n_train': int(len(train)), 'n_escape': int(y.sum()), 'n_subject': int((1-y).sum()),
        'n_target': int(len(target)), 'n_folds': cv.n_splits,
        'cv_roc_auc': float(roc_auc_score(y, heldout)),
        'cv_average_precision': float(average_precision_score(y, heldout)),
        'cv_balanced_accuracy_at_0_5': float(balanced_accuracy_score(y, heldout >= 0.5)),
        'escape_prevalence_baseline': float(y.mean()), 'features': list(features),
        'caveat': ('Gene-wise cross-validation is NOT cross-species validation. '
                   'Non-allelic female RNA/H3K27me3 features do not prove Xi escape. '
                   'Xist+ versus Xist- contrasts are not equivalent to Xi-versus-Xa expression. '
                   'Check tissue, cell type, batch and class balance before interpreting.'),
    }
    model.fit(train[list(features)], y)
    out = target[['gene', 'mus_gene']].copy()
    out['mus_literature_status'] = out.mus_gene.map(labels.set_index('mus_gene').xci_status)
    out['mus_label_source_doi'] = out.mus_gene.map(labels.set_index('mus_gene').source_doi)
    out['escape_like_probability'] = model.predict_proba(target[list(features)])[:, 1]
    out['prediction'] = np.select(
        [out.escape_like_probability >= 0.8, out.escape_like_probability <= 0.2],
        ['escape_like_candidate', 'subject_like_candidate'], default='uncertain')
    for f in features:
        out[f] = target[f].to_numpy()
    out = out.sort_values('escape_like_probability', ascending=False).reset_index(drop=True)
    heldout_df = train[['gene', 'xci_status', 'source_doi']].copy()
    heldout_df['heldout_escape_probability'] = heldout
    return out, metrics, heldout_df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['mus-female', 'pero-female', 'orthologs', 'labels', 'out', 'metrics', 'heldout']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--features', nargs='+', default=list(DEFAULT_FEATURES))
    args = p.parse_args()
    out, metrics, heldout = predict(
        pd.read_csv(args.mus_female), pd.read_csv(args.pero_female),
        pd.read_csv(args.orthologs), pd.read_csv(args.labels), features=args.features)
    for path in [args.out, args.metrics, args.heldout]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    heldout.to_csv(args.heldout, index=False)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
