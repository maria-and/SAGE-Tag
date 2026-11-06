"""CLI to export per-sample gene features and Xist contrasts from Seurat-exported matrices."""
import argparse
import pandas as pd
from .feature_engineering import load_export,gene_peak_map,build_gene_features,paired_contrasts

def main():
    p=argparse.ArgumentParser()
    for x in ['input-dir','species','annotation','out-features','out-contrasts']:p.add_argument('--'+x,required=True)
    a=p.parse_args(); meta,genes,peaks,rna,h3k=load_export(a.input_dir,a.species)
    ann=pd.read_csv(a.annotation,sep='\t'); links=gene_peak_map(ann,peaks)
    f=build_gene_features(meta,genes,rna,h3k,links,a.species)
    f.to_csv(a.out_features,index=False); paired_contrasts(f).to_csv(a.out_contrasts,index=False)
if __name__=='__main__':main()
