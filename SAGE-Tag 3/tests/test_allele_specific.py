from pipeline.allele_specific import allele_summary, genes_at, load_bed

def test_allele_summary():
    assert allele_summary(2, 1) == ('insufficient_coverage', None)
    assert allele_summary(5, 0) == ('one_allele_observed', 0)
    assert allele_summary(6, 4) == ('both_alleles_observed', 0.4)

def test_bed_coordinate_mapping(tmp_path):
    p = tmp_path / 'x.bed'
    p.write_text('chrX\t100\t200\tGeneA\nchrX\t200\t250\tGeneB\n')
    g = load_bed(p)
    assert genes_at(g, 'chrX', 100) == ['GeneA']
    assert genes_at(g, 'chrX', 199) == ['GeneA']
    assert genes_at(g, 'chrX', 200) == ['GeneB']
    assert genes_at(g, 'chrX', 99) == []
