#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro
from sklearn.impute import SimpleImputer
from fancyimpute import KNN
from fancyimpute import SoftImpute
from fancyimpute import IterativeImputer


# In[16]:


def nan(df):
    result= df.isna().sum()
    return result


# In[17]:


def shapiro(df):
    results= {}
    for col in df.select_dtypes(include=['int','float']).columns:
        stat,p= stats.shapiro(df[col])
        results[col]=stat,p
    return results


# In[18]:


def ks(df):
    results= {}
    for col in df.select_dtypes(include=['int','float']).columns:
        stat,p= stats.kstest(df[col],'norm')
        results[col]=stat,p
    return results


# In[19]:


def removenan(df):
    df= df.dropna().reset_index(drop=True)
    return df


# In[20]:


def imputer(df,strategy='mean'):
    
    imputation= SimpleImputer(strategy=strategy)
    
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    return df_imputed


# In[21]:


def knnimputer(df, method='knn'):
    df_num= df.select_dtypes(include=['int','float','uint'])
    df_cat= df.select_dtypes(exclude=['int','float','uint'])
    
    if method=='knn':
        imputer= KNN()
    elif method== 'soft':
        imputer= SoftImpute()
    elif method== 'iteractive':
        imputer= IterativeImputer
    else:
        raise ValueError("Metodo no soportado")
        
    imputed_num= imputer.fit_transform(df_num)
    imputed_cat= imputer.fit_transofmr(df_cat)
    
    df_imputed= pd.concat([imputed_num,imputed_cat],axis=1)
    
    return df_imputed


# In[ ]:




