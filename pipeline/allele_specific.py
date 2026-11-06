"""Allele-informative RNA counting for outbred single-nucleus data.

Requires coordinate-sorted, indexed BAM, indexed heterozygous VCF, and gene BED.
Outputs descriptive allelic counts, NOT definitive XCI escape calls.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

BASES = set('ACGT')


def allele_summary(ref_count, alt_count, min_total=5):
    total = ref_count + alt_count
    if total < min_total:
        return 'insufficient_coverage', None
    return ('both_alleles_observed' if ref_count > 0 and alt_count > 0 else 'one_allele_observed',
            min(ref_count, alt_count) / total)


def load_bed(path):
    """BED: chromosome, zero-based start, exclusive end, gene ID; optional strand."""
    genes = defaultdict(list)
    with open(path, newline='') as handle:
        for line in handle:
            if not line.strip() or line.startswith('#'): continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 4: raise ValueError('Gene BED needs at least 4 tab-delimited columns')
            chrom, start, end, gene = parts[:4]
            start, end = int(start), int(end)
            if start < 0 or end <= start: raise ValueError('Invalid BED coordinates')
            genes[chrom].append((start, end, gene))
    return genes


def genes_at(genes, chrom, zero_based_pos):
    return sorted({name for start, end, name in genes.get(chrom, []) if start <= zero_based_pos < end})


def count_alleles(bam_path, vcf_path, gene_bed, output_prefix, sample, cell_tag='CB', umi_tag='UB',
                  min_mapq=30, min_baseq=20, min_gene_molecules=5, allow_no_cell_tag=False):
    try:
        import pysam
    except ImportError as e:
        raise RuntimeError('Install pysam: python -m pip install pysam') from e
    genes = load_bed(gene_bed)
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    variant_rows = []
    gene_counts = defaultdict(lambda: [0, 0, 0])  # ref, alt, informative sites
    diagnostics = defaultdict(int)
    with pysam.AlignmentFile(bam_path, 'rb') as bam, pysam.VariantFile(vcf_path) as vcf:
        if sample not in list(vcf.header.samples):
            raise ValueError(f'Sample {sample!r} not found in VCF samples: {list(vcf.header.samples)}')
        if not bam.has_index(): raise ValueError('BAM must be coordinate-sorted and indexed')
        try:
            iterator = vcf.fetch()
        except (ValueError, OSError) as e:
            raise ValueError('VCF must be bgzip-compressed and tabix-indexed') from e
        for variant in iterator:
            diagnostics['variants_seen'] += 1
            if variant.contig not in genes or variant.contig not in bam.references: continue
            if len(variant.ref) != 1 or variant.ref not in BASES or not variant.alts or len(variant.alts) != 1 or len(variant.alts[0]) != 1 or variant.alts[0] not in BASES:
                continue
            gt = variant.samples[sample].get('GT')
            if gt is None or len(gt) != 2 or set(gt) != {0, 1}: continue
            overlapping = genes_at(genes, variant.contig, variant.start)
            # Exclude overlapping genes to avoid ambiguous assignment.
            if len(overlapping) != 1:
                diagnostics['ambiguous_or_unannotated_sites'] += 1
                continue
            gene = overlapping[0]
            diagnostics['heterozygous_gene_snps'] += 1
            # Deduplicate per allele, cell, variant, UMI. For no UMI use read name.
            observed = set()
            per_cell = defaultdict(lambda: [0, 0])
            for column in bam.pileup(variant.contig, variant.start, variant.start + 1,
                                     truncate=True, stepper='nofilter', min_base_quality=0,
                                     max_depth=1000000):
                if column.reference_pos != variant.start: continue
                for p in column.pileups:
                    read = p.alignment
                    if p.is_del or p.is_refskip or p.query_position is None: continue
                    if read.is_unmapped or read.is_secondary or read.is_supplementary or read.is_duplicate or read.is_qcfail or read.mapping_quality < min_mapq: continue
                    qpos = p.query_position
                    if read.query_qualities is None or read.query_qualities[qpos] < min_baseq: continue
                    if not read.has_tag(cell_tag):
                        diagnostics['reads_without_cell_tag'] += 1
                        if not allow_no_cell_tag: continue
                        cell = 'BULK'
                    else:
                        cell = read.get_tag(cell_tag)
                    base = read.query_sequence[qpos].upper()
                    if base == variant.ref: allele = 0
                    elif base == variant.alts[0]: allele = 1
                    else: continue
                    molecule = read.get_tag(umi_tag) if read.has_tag(umi_tag) else read.query_name
                    key = (cell, molecule, allele)
                    if key in observed: continue
                    observed.add(key)
                    per_cell[cell][allele] += 1
            for cell, (ref_n, alt_n) in per_cell.items():
                if ref_n + alt_n == 0: continue
                variant_rows.append(dict(chrom=variant.contig, pos_1based=variant.pos,
                                         gene=gene, cell=cell, ref=variant.ref,
                                         alt=variant.alts[0], ref_molecules=ref_n,
                                         alt_molecules=alt_n, phased=bool(variant.samples[sample].phased)))
                entry = gene_counts[(cell, gene)]
                entry[0] += ref_n; entry[1] += alt_n; entry[2] += 1
    variant_file = str(output_prefix) + '_variant_alleles.csv'
    gene_file = str(output_prefix) + '_gene_alleles.csv'
    with open(variant_file, 'w', newline='') as f:
        fields = ['chrom','pos_1based','gene','cell','ref','alt','ref_molecules','alt_molecules','phased']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(variant_rows)
    with open(gene_file, 'w', newline='') as f:
        fields = ['cell','gene','ref_molecules','alt_molecules','informative_sites','minor_allele_fraction','status']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for (cell, gene), (ref_n, alt_n, sites) in sorted(gene_counts.items()):
            status, fraction = allele_summary(ref_n, alt_n, min_gene_molecules)
            writer.writerow(dict(cell=cell, gene=gene, ref_molecules=ref_n,
                                 alt_molecules=alt_n, informative_sites=sites,
                                 minor_allele_fraction='' if fraction is None else fraction,
                                 status=status))
    print('Wrote:', variant_file, gene_file)
    print('Diagnostics:', dict(diagnostics))
    if not variant_rows:
        print('WARNING: No informative allele observations. Check BAM/VCF contig naming, SNP coverage, and CB tags.')
    print('IMPORTANT: REF/ALT is NOT a consistent haplotype across SNPs. Do not interpret gene totals as Xi/Xa without phasing.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bam', required=True, help='RNA BAM, sorted and indexed; CB and UB tags preferred')
    parser.add_argument('--vcf', required=True, help='Indexed bgzip VCF with heterozygous sample genotypes')
    parser.add_argument('--sample', required=True, help='VCF sample ID')
    parser.add_argument('--genes-bed', required=True, help='BED4+ with X-linked gene intervals')
    parser.add_argument('--out-prefix', required=True)
    parser.add_argument('--cell-tag', default='CB')
    parser.add_argument('--umi-tag', default='UB')
    parser.add_argument('--min-mapq', type=int, default=30)
    parser.add_argument('--min-baseq', type=int, default=20)
    parser.add_argument('--min-gene-molecules', type=int, default=5)
    parser.add_argument('--allow-no-cell-tag', action='store_true', help='Count untagged reads in a single BULK group; NOT single-cell evidence')
    a = parser.parse_args()
    count_alleles(a.bam, a.vcf, a.genes_bed, a.out_prefix, a.sample, a.cell_tag,
                  a.umi_tag, a.min_mapq, a.min_baseq, a.min_gene_molecules, a.allow_no_cell_tag)

if __name__ == '__main__': main()
