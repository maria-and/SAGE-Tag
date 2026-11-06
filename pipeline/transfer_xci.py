"""Literature-supervised Mus XCI-status model -> Peromyscus ortholog prediction.
IMPORTANT: This predicts similarity to Mus XCI status, NOT causal Xist repression.
Input Mus features must NOT be computed using literature status labels.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, average_precision_score, roc_auc_score

FEATURES=['rna_mean','h3k_mean','rna_log2fc','h3k_log2fc']


def prepare(mus,pero,orthologs,labels):
    """Tables must have one row per gene; orthologs: mus_gene, pero_gene (unique one-to-one).
    labels: mus_gene, xci_status in {'subject','escape'}; source DOI required.
    """
    for cols,df in [(['gene'],mus),(['gene'],pero),(['mus_gene','pero_gene'],orthologs),(['mus_gene','xci_status','source_doi'],labels)]:
        if not set(cols).issubset(df): raise ValueError(f'Missing columns {set(cols)-set(df.columns)}')
    if orthologs.mus_gene.duplicated().any() or orthologs.pero_gene.duplicated().any():
        raise ValueError('Use only unique one-to-one orthologs')
    if labels.mus_gene.duplicated().any(): raise ValueError('Duplicate literature labels')
    if not labels.xci_status.isin(['subject','escape']).all(): raise ValueError('Unknown XCI label')
    if labels.source_doi.isna().any(): raise ValueError('Every literature label requires provenance')
    train=mus.merge(labels,left_on='gene',right_on='mus_gene',validate='one_to_one')
    target=pero.merge(orthologs,left_on='gene',right_on='pero_gene',validate='one_to_one')
    return train,target


def run(mus,pero,orthologs,labels,features=FEATURES):
    train,target=prepare(mus,pero,orthologs,labels)
    if len(train)<20 or train.xci_status.value_counts().min()<5:
        raise ValueError('Insufficient labeled examples: require >=20 genes and >=5 per class')
    for col in features:
        if col not in train or col not in target: raise ValueError(f'Missing feature: {col}')
    y=(train.xci_status=='subject').astype(int)
    model=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=False)),('scale',StandardScaler()),('clf',LogisticRegression(class_weight='balanced',max_iter=3000,random_state=42))])
    # Gene-level grouping avoids leakage if a gene is ever represented more than once.
    # A single row per gene is enforced in the current implementation.
    if train.gene.duplicated().any() or target.gene.duplicated().any():
        raise ValueError('Aggregate sample-level features to one row per gene before model training')
    n_splits=min(5,int(y.value_counts().min()))
    cv=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=42)
    pred=cross_val_predict(model,train[features],y,groups=train.gene,cv=cv,method='predict_proba')[:,1]
    metrics={'n_train':int(len(train)),'n_subject':int(y.sum()),'n_escape':int((1-y).sum()),'cv_roc_auc':float(roc_auc_score(y,pred)),'cv_average_precision':float(average_precision_score(y,pred)),'cv_balanced_accuracy':float(balanced_accuracy_score(y,pred>=0.5)),'features':features,'warning':'Gene-wise CV only; cross-species generalization unvalidated. Avoid using same study for feature construction and evaluation.'}
    model.fit(train[features],y)
    result=target[['gene','mus_gene']].copy()
    result['mus_xci_subject_similarity']=model.predict_proba(target[features])[:,1]
    result['prediction']='uncertain'
    result.loc[result.mus_xci_subject_similarity>=0.8,'prediction']='subject_like'
    result.loc[result.mus_xci_subject_similarity<=0.2,'prediction']='escape_like'
    # Provide measured features alongside predictions for interpretability.
    for f in features: result[f]=target[f].values
    return result.sort_values('mus_xci_subject_similarity',ascending=False),metrics


def main():
    p=argparse.ArgumentParser()
    for x in ['mus','pero','orthologs','labels','out','metrics']:p.add_argument('--'+x,required=True)
    a=p.parse_args()
    result,metrics=run(*(pd.read_csv(getattr(a,x)) for x in ['mus','pero','orthologs','labels']))
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(a.out,index=False); Path(a.metrics).write_text(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
