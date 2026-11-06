"""Aggregate sample-level Xist contrasts to one row per gene for exploratory transfer modeling.
Must stratify by sex and (if available) cell type before use.
"""
import argparse
import pandas as pd

def aggregate(features,contrasts,sex):
    f=features[features.sex==sex].copy(); c=contrasts[contrasts.sex==sex].copy()
    if f.empty or c.empty: raise ValueError('No matching sex or no paired Xist groups')
    # Reference state: Xist-negative cells, with paired sample-level contrasts.
    base=f[f.xist_positive.astype(str).str.lower()=='false'].groupby('gene',as_index=False).agg(rna_mean=('rna_mean','mean'),h3k_mean=('h3k_mean','mean'))
    delta=c.groupby('gene',as_index=False).agg(rna_log2fc=('rna_log2fc','mean'),h3k_log2fc=('h3k_log2fc','mean'),n_samples=('sample','nunique'))
    return base.merge(delta,on='gene')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ['features','contrasts','sex','out']:p.add_argument('--'+arg,required=True)
    a=p.parse_args();aggregate(pd.read_csv(a.features),pd.read_csv(a.contrasts),a.sex).to_csv(a.out,index=False)
