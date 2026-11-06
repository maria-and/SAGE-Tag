"""Ortholog-aligned comparisons of Xist+ vs Xist- RNA/H3K27me3 contrasts."""
import argparse
import pandas as pd
import numpy as np


def compare(contrasts,orthologs):
    required={'species','sex','sample','gene','rna_log2fc','h3k_log2fc'}
    if not required.issubset(contrasts): raise ValueError(f'Missing {required-set(contrasts.columns)}')
    # Descriptive sample-level averages. Do not report cell-level p-values as biological-replicate tests.
    grouped=contrasts.groupby(['species','sex','gene'],as_index=False).agg(rna_log2fc=('rna_log2fc','mean'),h3k_log2fc=('h3k_log2fc','mean'),n_samples=('sample','nunique'))
    mus=grouped[grouped.species=='Mus'].rename(columns={'gene':'mus_gene','rna_log2fc':'mus_rna_log2fc','h3k_log2fc':'mus_h3k_log2fc','n_samples':'mus_samples'}).drop(columns='species')
    pero=grouped[grouped.species=='Pero'].rename(columns={'gene':'pero_gene','rna_log2fc':'pero_rna_log2fc','h3k_log2fc':'pero_h3k_log2fc','n_samples':'pero_samples'}).drop(columns='species')
    out=orthologs.merge(mus,on='mus_gene').merge(pero,on=['pero_gene','sex'])
    out['delta_rna_pero_minus_mus']=out.pero_rna_log2fc-out.mus_rna_log2fc
    out['delta_h3k_pero_minus_mus']=out.pero_h3k_log2fc-out.mus_h3k_log2fc
    out['pero_repression_signature']=(out.pero_rna_log2fc<0)&(out.pero_h3k_log2fc>0)
    out['mus_repression_signature']=(out.mus_rna_log2fc<0)&(out.mus_h3k_log2fc>0)
    out['signature_class']=np.select([out.pero_repression_signature&out.mus_repression_signature,out.pero_repression_signature&~out.mus_repression_signature,~out.pero_repression_signature&out.mus_repression_signature],['both','pero_only','mus_only'],default='neither')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--contrasts',required=True);p.add_argument('--orthologs',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();compare(pd.read_csv(a.contrasts),pd.read_csv(a.orthologs)).to_csv(a.out,index=False)
