"""Build gene-level tables from paired RNA/H3K27me3 counts, without treating cells as biological replicates."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


def load_export(folder, species):
    p = Path(folder)
    meta = pd.read_csv(p / f'{species}_metadata.csv').set_index('cell')
    genes = (p / f'{species}_genes.txt').read_text().splitlines()
    cells = (p / f'{species}_cells.txt').read_text().splitlines()
    peaks = (p / f'{species}_peaks.txt').read_text().splitlines()
    peak_cells = (p / f'{species}_peak_cells.txt').read_text().splitlines()
    if cells != peak_cells:
        raise ValueError('RNA and peak columns differ; explicit barcode alignment required')
    if set(cells) != set(meta.index):
        raise ValueError('Metadata barcodes do not match counts')
    meta = meta.loc[cells]
    rna = sparse.csr_matrix(mmread(p / f'{species}_rna.mtx'))
    h3k = sparse.csr_matrix(mmread(p / f'{species}_h3k.mtx'))
    if rna.shape != (len(genes),len(cells)) or h3k.shape != (len(peaks),len(cells)):
        raise ValueError('Matrix dimensions do not match gene/peak/barcode files')
    return meta, genes, peaks, rna, h3k


def parse_peaks(peaks):
    import re
    records = []
    for s in peaks:
        m = re.match(r'^(.*?)[-:](\d+)-(\d+)$', s)
        if not m: raise ValueError(f'Unrecognized peak coordinate: {s}')
        records.append((m.group(1).replace('chr','',1) if m.group(1).startswith('chr') else m.group(1),int(m.group(2)),int(m.group(3))))
    return pd.DataFrame(records,columns=['chromosome','start','end'])


def gene_peak_map(annotations, peaks, window=2000):
    """Gene TSS and peak coordinates must use the SAME reference genome and coordinate convention.
    annotations: gene, chromosome, start, end, strand. 1-based inclusive coordinates.
    """
    p = parse_peaks(peaks)
    a = annotations.copy()
    a['chromosome'] = a['chromosome'].astype(str).str.replace(r'^chr','',regex=True)
    a['tss'] = np.where(a['strand']=='-',a['end'],a['start'])
    links = {}
    for i,row in a.iterrows():
        match = (p.chromosome==row.chromosome)&(p.end>=row.tss-window)&(p.start<=row.tss+window)
        links[str(row.gene)] = np.flatnonzero(match.to_numpy())
    return links


def build_gene_features(meta, genes, rna, h3k, links, species, min_cells=10):
    """Within sex and sample, compare Xist+ vs Xist-. Values are descriptive, not inferential p-values.
    H3K CPM denominator uses ALL peaks, never just TSS-overlapping peaks.
    """
    rna_depth = np.asarray(rna.sum(axis=0)).ravel()
    h3k_depth = np.asarray(h3k.sum(axis=0)).ravel()
    rna_norm = rna.multiply(1e4/np.maximum(rna_depth,1)).tocsr()
    rna_norm.data = np.log1p(rna_norm.data)
    h3k_cpm = h3k.multiply(1e6/np.maximum(h3k_depth,1)).tocsr()
    idx = {g:i for i,g in enumerate(genes)}
    rows = []
    for (sex,sample),grp in meta.groupby(['sex','sample'],dropna=False):
        for status in [True,False]:
            ids = np.flatnonzero((meta['sex'].to_numpy()==sex)&(meta['sample'].to_numpy()==sample)&(meta['xist_positive'].astype(bool).to_numpy()==status))
            if len(ids)<min_cells: continue
            for gene, pk in links.items():
                if gene not in idx: continue
                rna_vals = rna_norm[idx[gene],ids].toarray().ravel()
                h3k_vals = np.asarray(h3k_cpm[pk,:][:,ids].sum(axis=0)).ravel() if len(pk) else np.array([])
                rows.append(dict(species=species,sex=sex,sample=sample,xist_positive=status,gene=gene,n_cells=len(ids),rna_mean=float(rna_vals.mean()),h3k_mean=float(h3k_vals.mean()) if len(h3k_vals) else np.nan))
    return pd.DataFrame(rows)


def paired_contrasts(features):
    keys=['species','sex','sample','gene']
    pos=features[features.xist_positive==True].drop(columns='xist_positive').rename(columns={'n_cells':'n_pos','rna_mean':'rna_pos','h3k_mean':'h3k_pos'})
    neg=features[features.xist_positive==False].drop(columns='xist_positive').rename(columns={'n_cells':'n_neg','rna_mean':'rna_neg','h3k_mean':'h3k_neg'})
    out=pos.merge(neg,on=keys,validate='one_to_one')
    out['rna_log2fc']=np.log2((out.rna_pos+1)/(out.rna_neg+1))
    out['h3k_log2fc']=np.log2((out.h3k_pos+1)/(out.h3k_neg+1))
    return out
