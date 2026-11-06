import unittest
import pandas as pd
from pipeline.transfer_xci import run
from pipeline.species_comparison import compare

class TestPipeline(unittest.TestCase):
    def test_transfer(self):
        genes=[f'g{i}' for i in range(40)]
        mus=pd.DataFrame({'gene':genes,'rna_mean':[i/10 for i in range(40)],'h3k_mean':[40-i for i in range(40)],'rna_log2fc':[i/40 for i in range(40)],'h3k_log2fc':[1-i/40 for i in range(40)]})
        pero=mus.copy();pero['gene']=[f'p{i}' for i in range(40)]
        orth=pd.DataFrame({'mus_gene':genes,'pero_gene':pero.gene})
        labels=pd.DataFrame({'mus_gene':genes,'xci_status':['subject' if i<20 else 'escape' for i in range(40)],'source_doi':['10.1234/example']*40})
        result,metrics=run(mus,pero,orth,labels)
        self.assertEqual(len(result),40)
        self.assertTrue(result.mus_xci_subject_similarity.between(0,1).all())
        self.assertIn('cv_roc_auc',metrics)
    def test_comparison(self):
        data=pd.DataFrame([{'species':s,'sex':'Male','sample':s+'1','gene':g,'rna_log2fc':-1,'h3k_log2fc':1} for s,g in [('Mus','g'),('Pero','p')]])
        out=compare(data,pd.DataFrame({'mus_gene':['g'],'pero_gene':['p']}))
        self.assertEqual(out.signature_class.iloc[0],'both')
if __name__=='__main__':unittest.main()
