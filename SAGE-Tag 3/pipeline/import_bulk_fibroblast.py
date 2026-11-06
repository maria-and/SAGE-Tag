"""Import fibroblast bulk TPM/RPKM matrices without merging species by gene ID.

Outputs tissue-specific female feature tables and descriptive male contrasts.
The RNA and CUT&Tag samples are NOT assumed to be paired animals.
RPKM is an enrichment proxy, not normalized ChIP/CUT&Tag occupancy.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SPEC = {
 'Mus': {'h3k': {'Male':['Mus_M_1','Mus_M_2'], 'Female':['Mus_F_1','Mus_F_2']},
         'rna': {'Male':['M1','M2'], 'Female':['F1','F2','F3']}},
 'Pero': {'h3k': {'Male':['Peromyscus_M_1','Peromyscus_M_2'], 'Female':['Peromyscus_F_1','Peromyscus_F_2']},
          'rna': {'Male':['M1','M2'], 'Female':['F1','F2','F3']}}
}

def read_matrix(path, id_col, groups):
    df = pd.read_csv(path)
    needed = [id_col] + sum(groups.values(), [])
    absent = set(needed) - set(df.columns)
    if absent: raise ValueError(f'{path}: missing {sorted(absent)}')
    out = df[needed].rename(columns={id_col:'gene'}).copy()
    out['gene'] = out['gene'].astype('string').str.strip()
    out = out.dropna(subset=['gene'])
    if out.gene.duplicated().any():
        raise ValueError(f'{path}: duplicate gene identifiers; resolve before importing')
    for col in needed[1:]:
        out[col] = pd.to_numeric(out[col], errors='raise')
        if (out[col].dropna() < 0).any(): raise ValueError(f'{path}: negative values in {col}')
    return out

def summarize(df, groups, prefix):
    result = df[['gene']].copy()
    for sex, cols in groups.items():
        vals = df[cols].to_numpy(dtype=float)
        result[f'{prefix}_{sex.lower()}_mean'] = np.nanmean(vals, axis=1)
        result[f'{prefix}_{sex.lower()}_n'] = np.isfinite(vals).sum(axis=1)
    return result

def build_features(h3k_path, rna_path, species, chromosome_col=None, x_only=True):
    groups = SPEC[species]
    h = read_matrix(h3k_path, 'gene_id', groups['h3k'])
    r = read_matrix(rna_path, 'Name', groups['rna'])
    raw_h = pd.read_csv(h3k_path, usecols=lambda c: c in ['gene_id','chromosome.x','chromosome'])
    if x_only:
        col = chromosome_col or next((c for c in ['chromosome.x','chromosome'] if c in raw_h), None)
        if col is None: raise ValueError(f'{species}: chromosome annotation missing in H3K table')
        if raw_h.gene_id.duplicated().any(): raise ValueError('Duplicate IDs in chromosome table')
        x_ids = set(raw_h.loc[raw_h[col].astype(str).isin(['X','chrX','NC_051083.1','NC-051083.1']), 'gene_id'].astype(str))
        h = h[h.gene.astype(str).isin(x_ids)]
    hs = summarize(h, groups['h3k'], 'h3k')
    rs = summarize(r, groups['rna'], 'rna')
    allgenes = hs.merge(rs, on='gene', how='inner', validate='one_to_one')
    if allgenes.empty: raise ValueError(f'{species}: no matching RNA/H3K gene IDs within species; inspect identifiers')
    for sex in ['female','male']:
        # Features are within-species gene-wise summaries, NOT cross-species fold changes.
        t = allgenes[['gene',f'rna_{sex}_mean',f'h3k_{sex}_mean',f'rna_{sex}_n',f'h3k_{sex}_n']].copy()
        t['rna_mean'] = np.log1p(t[f'rna_{sex}_mean'])
        t['h3k_mean'] = np.log1p(t[f'h3k_{sex}_mean'])
        # Sex contrast is an exploratory proxy, not Xist+/Xist- or allele-specific XCI.
        t['rna_log2fc'] = np.log2((allgenes[f'rna_{sex}_mean'] + .1)/(allgenes[f'rna_{"male" if sex == "female" else "female"}_mean'] + .1))
        t['h3k_log2fc'] = np.log2((allgenes[f'h3k_{sex}_mean'] + .1)/(allgenes[f'h3k_{"male" if sex == "female" else "female"}_mean'] + .1))
        t['contrast'] = f'{sex}_vs_{"male" if sex == "female" else "female"}'
        t['tissue'] = 'fibroblast'
        t['species'] = species
        yield sex, t

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mus-h3k', required=True); p.add_argument('--pero-h3k', required=True)
    p.add_argument('--mus-tpm', required=True); p.add_argument('--pero-tpm', required=True)
    p.add_argument('--out-dir', default='outputs')
    p.add_argument('--all-genes', action='store_true', help='Do not restrict to X chromosome')
    a=p.parse_args(); out=Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for sp,h,r in [('Mus',a.mus_h3k,a.mus_tpm),('Pero',a.pero_h3k,a.pero_tpm)]:
        for sex,df in build_features(h,r,sp,x_only=not a.all_genes):
            name=f'fibro_{sp.lower()}_{sex}.csv'
            df.to_csv(out/name,index=False)
            print(f'{name}: {len(df)} genes')
    print('Note: use ortholog mapping for cross-species joins; no animal matching is assumed.')
if __name__=='__main__': main()
