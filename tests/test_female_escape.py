import numpy as np
import pandas as pd
import pytest
from pipeline.predict_female_escape import predict


def fixture():
    rng = np.random.default_rng(7)
    n = 50
    labels = np.array(['escape'] * 10 + ['subject'] * 40)
    genes = [f'm{i}' for i in range(n)]
    mus = pd.DataFrame({'gene': genes, 'rna_mean': rng.normal(0, 1, n) + (labels == 'escape') * 2,
                        'h3k_mean': rng.normal(0, 1, n) - (labels == 'escape') * 2,
                        'rna_log2fc': rng.normal(0, 1, n), 'h3k_log2fc': rng.normal(0, 1, n)})
    pero = mus.copy()
    pero['gene'] = [f'p{i}' for i in range(n)]
    orth = pd.DataFrame({'mus_gene': genes, 'pero_gene': pero.gene})
    lit = pd.DataFrame({'mus_gene': genes, 'xci_status': labels, 'source_doi': ['10.example/test'] * n})
    return mus, pero, orth, lit


def test_predict_shapes_and_probabilities():
    result, metrics, heldout = predict(*fixture())
    assert len(result) == 50
    assert result.escape_like_probability.between(0, 1).all()
    assert len(heldout) == 50
    assert metrics['n_escape'] == 10
    assert metrics['escape_prevalence_baseline'] == .2
    assert 'mus_literature_status' in result


def test_duplicate_ortholog_rejected():
    mus, pero, orth, lit = fixture()
    orth.loc[1, 'pero_gene'] = orth.loc[0, 'pero_gene']
    with pytest.raises(ValueError, match='unique'):
        predict(mus, pero, orth, lit)


def test_insufficient_escape_examples_rejected():
    mus, pero, orth, lit = fixture()
    lit.loc[4:, 'xci_status'] = 'subject'
    with pytest.raises(ValueError, match='at least'):
        predict(mus, pero, orth, lit)
