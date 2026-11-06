import numpy as np
import pandas as pd
import pytest
from pipeline.cross_tissue_xci import integrate


def fixture():
    rng = np.random.default_rng(11)
    n = 50
    genes = [f'm{i}' for i in range(n)]
    status = ['escape'] * 12 + ['subject'] * 38
    mus = pd.DataFrame({'gene': genes,
                        'rna_mean': rng.normal(size=n) + np.array([s == 'escape' for s in status]),
                        'h3k_mean': rng.normal(size=n),
                        'rna_log2fc': rng.normal(size=n),
                        'h3k_log2fc': rng.normal(size=n)})
    pero = mus.copy()
    pero.gene = [f'p{i}' for i in range(n)]
    orth = pd.DataFrame({'mus_gene': genes, 'pero_gene': pero.gene})
    labels = pd.DataFrame({'mus_gene': genes, 'xci_status': status, 'source_doi': ['10.test/doi'] * n})
    return (mus, pero), (mus.copy(), pero.copy()), orth, labels


def test_two_tissue_scores_and_male():
    liver, fibro, orth, labels = fixture()
    male = pd.DataFrame({'gene': ['p0', 'p1'], 'rna_log2fc': [-1, 1], 'h3k_log2fc': [1, -1]})
    out, metrics = integrate(liver, fibro, orth, labels, male, male)
    assert len(out) == 50
    assert out.n_tissues_scored.eq(2).all()
    assert out.mean_escape_like_score.between(0, 1).all()
    assert out.set_index('pero_gene').loc['p0', 'male_signature_class'] == 'both'
    assert out.set_index('pero_gene').loc['p1', 'male_signature_class'] == 'neither'
    assert 'cv_roc_auc' in metrics['liver']


def test_single_tissue_and_missing_male():
    liver, _, orth, labels = fixture()
    out, _ = integrate(liver, None, orth, labels)
    assert out.escape_evidence_class.eq('single_tissue_only').all()
    assert out.male_signature_class.eq('missing_or_uncertain').all()


def test_rejects_duplicate_orthologs():
    liver, fibro, orth, labels = fixture()
    orth.loc[1, 'pero_gene'] = orth.loc[0, 'pero_gene']
    with pytest.raises(ValueError, match='unique'):
        integrate(liver, fibro, orth, labels)
