import pandas as pd
import pytest
from pipeline.import_bulk_fibroblast import build_features
from pipeline.uce_bridge import attach

def test_bulk_three_female_replicates(tmp_path):
    h=pd.DataFrame({'gene_id':['g1','g2'],'chromosome.x':['X','1'],
        'Mus_M_1':[2,1],'Mus_M_2':[2,1],'Mus_F_1':[4,1],'Mus_F_2':[6,1]})
    r=pd.DataFrame({'Name':['g1','g2'],'M1':[3,1],'M2':[3,1],
                    'F1':[2,1],'F2':[4,1],'F3':[9,1]})
    hp=tmp_path/'h.csv';rp=tmp_path/'r.csv';h.to_csv(hp,index=False);r.to_csv(rp,index=False)
    data=dict(build_features(hp,rp,'Mus'))
    assert len(data['female'])==1
    assert data['female'].iloc[0].rna_female_mean==5
    assert data['female'].iloc[0].rna_female_n==3

def test_uce_alignment():
    m=pd.DataFrame({'cell_id':['c2','c1'],'species':['Mus','Pero']})
    e=pd.DataFrame({'cell_id':['c1','c2'],'uce_0':[1.,2.]})
    assert attach(m,e).uce_0.tolist()==[2.,1.]
    with pytest.raises(ValueError): attach(m,e.iloc[:1])
