#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from scipy.stats import shapiro
from sklearn.impute import SimpleImputer


# In[7]:


def nan(df):
    result= df.isna().sum()
    return result


# In[8]:


def shapiro(df):
    results= {}
    for col in df.select_dtypes(include=['int','float']).columns:
        stat,p= stats.shapiro(df[col])
        results[col]=stat,p
    return results


# In[9]:


def ks(df):
    results= {}
    for col in df.select_dtypes(include=['int','float']).columns:
        stat,p= stats.kstest(df[col],'norm')
        results[col]=stat,p
    return results


# In[10]:


def removenan(df):
    df= df.dropna().reset_index(drop=True)
    return df


# In[11]:


def imputer(df,strategy='mean'):
    
    imputation= SimpleImputer(strategy=strategy)
    
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    return df_imputed


# In[ ]:




