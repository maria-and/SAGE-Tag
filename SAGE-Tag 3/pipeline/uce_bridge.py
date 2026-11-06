"""Validate and attach externally computed UCE cell embeddings to Paired-Tag metadata.

UCE inference itself requires the official UCE environment/model weights; this
bridge intentionally does not pretend to implement or run the foundation model.
Use gene-expression counts only, map supported orthologs, and verify species
coverage before generating embeddings with the official UCE workflow.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def attach(metadata, embeddings):
    for name, df in [('metadata',metadata),('embeddings',embeddings)]:
        if 'cell_id' not in df: raise ValueError(f'{name}: missing cell_id')
        if df.cell_id.isna().any() or df.cell_id.duplicated().any():
            raise ValueError(f'{name}: cell_id must be unique and nonmissing')
    numeric = [c for c in embeddings if c.startswith('uce_')]
    if not numeric: raise ValueError('Embeddings require numeric uce_0, uce_1, ... columns')
    if not all(pd.api.types.is_numeric_dtype(embeddings[c]) for c in numeric):
        raise ValueError('UCE columns must be numeric')
    if not np.isfinite(embeddings[numeric].to_numpy(dtype=float)).all():
        raise ValueError('UCE embeddings contain nonfinite values')
    merged=metadata.merge(embeddings[['cell_id']+numeric],on='cell_id',how='left',validate='one_to_one',indicator=True)
    if (merged._merge != 'both').any():
        raise ValueError(f'{(merged._merge != "both").sum()} metadata cells lack UCE embeddings')
    return merged.drop(columns='_merge')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',required=True); p.add_argument('--embeddings',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); merged=attach(pd.read_csv(a.metadata),pd.read_csv(a.embeddings))
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); merged.to_csv(a.out,index=False)
    print(f'Attached UCE embeddings to {len(merged)} cells')
if __name__=='__main__':main()
