"""SAGE-Tag command-line entry point. Run `python sage.py --help`."""
import argparse
import json
from pathlib import Path
import pandas as pd
from pipeline.import_bulk_fibroblast import build_features
from pipeline.cross_tissue_xci import integrate
from pipeline.uce_bridge import attach


def bulk(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for species, h3k, rna in [('Mus', args.mus_h3k, args.mus_tpm), ('Pero', args.pero_h3k, args.pero_tpm)]:
        for sex, frame in build_features(h3k, rna, species, x_only=not args.all_genes):
            name = f'fibro_{species.lower()}_{sex}.csv'
            frame.to_csv(out / name, index=False)
            summary[name] = len(frame)
            print(f'{out/name}: {len(frame)} genes')
    (out/'bulk_import_summary.json').write_text(json.dumps(summary, indent=2))


def integrate_cmd(args):
    def read(path, required):
        frame = pd.read_csv(path)
        missing = set(required) - set(frame.columns)
        if missing: raise ValueError(f'{path}: missing columns {sorted(missing)}')
        return frame
    def pair(mus, pero, tissue):
        if bool(mus) != bool(pero): raise ValueError(f'{tissue}: supply both Mus and Peromyscus female files')
        return (read(mus,['gene']), read(pero,['gene'])) if mus else None
    liver = pair(args.liver_mus_female,args.liver_pero_female,'liver')
    fibro = pair(args.fibro_mus_female,args.fibro_pero_female,'fibroblast')
    orth = read(args.orthologs,['mus_gene','pero_gene'])
    labels = read(args.labels,['mus_gene','xci_status','source_doi'])
    male_req=['gene','rna_log2fc','h3k_log2fc']
    result, metrics = integrate(liver, fibro, orth, labels,
        read(args.liver_pero_male,male_req) if args.liver_pero_male else None,
        read(args.fibro_pero_male,male_req) if args.fibro_pero_male else None)
    # Preserve the meaning of distinct contrasts; they must not be interpreted as equivalent.
    metrics['male_contrast_definitions'] = {
        'fibroblast': 'Peromyscus male versus Peromyscus female (bulk RNA/CUT&Tag)',
        'liver': 'User-supplied contrast; verify whether Xist+ versus Xist- or another definition'
    }
    for path in (args.out, args.metrics): Path(path).parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.out,index=False)
    Path(args.metrics).write_text(json.dumps(metrics,indent=2))
    print(f'Wrote {len(result)} ortholog-mapped candidates to {args.out}')


def uce(args):
    frame=attach(pd.read_csv(args.metadata),pd.read_csv(args.embeddings))
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(args.out,index=False)
    print(f'Attached externally generated UCE embeddings to {len(frame)} cells')


def main():
    p=argparse.ArgumentParser(description='SAGE-Tag: tissue-aware, cross-species XCI analysis')
    sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('bulk',help='Prepare fibroblast gene features from TPM/RPKM CSVs')
    for name in ['mus-h3k','pero-h3k','mus-tpm','pero-tpm']: b.add_argument('--'+name,required=True)
    b.add_argument('--out-dir',default='outputs'); b.add_argument('--all-genes',action='store_true')
    b.set_defaults(func=bulk)
    i=sub.add_parser('integrate',help='Train tissue-specific female models and rank orthologs')
    for name in ['orthologs','labels','out','metrics']: i.add_argument('--'+name,required=True)
    for name in ['liver-mus-female','liver-pero-female','fibro-mus-female','fibro-pero-female','liver-pero-male','fibro-pero-male']:
        i.add_argument('--'+name)
    i.set_defaults(func=integrate_cmd)
    u=sub.add_parser('uce-attach',help='Join externally computed UCE embeddings to liver metadata')
    for name in ['metadata','embeddings','out']: u.add_argument('--'+name,required=True)
    u.set_defaults(func=uce)
    a=p.parse_args(); a.func(a)
if __name__=='__main__': main()
