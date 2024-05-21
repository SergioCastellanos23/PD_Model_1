#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from optbinning import OptimalBinning
from varclushi import VarClusHi
from scipy import stats
import statsmodels.api as sm


# In[3]:


def contingency(df, y, rot=0,show_legend=False):
    cat_col= df.select_dtypes(include=['object']).columns
    plots=[]
    
    for column in cat_col:
        table = pd.crosstab(df[column], y)
        plot = table.div(table.sum(1).replace(0,1), axis=0).plot(kind='barh', stacked=True, legend=show_legend, rot=rot)
        plots.append(plot)
    
    return plots


# In[4]:

def olstest(df, y):
    df= df.astype('int')
    results = {}
    
    for column in df.columns:
        if column == y:
            continue
        
        x = df[[column]]
        y_col = df[[y]]
        
        data = pd.concat([x, y_col], axis=1)
        data.columns = ['x', 'y']
        
        ols_model = sm.OLS(data['y'], sm.add_constant(data['x'])).fit()
        
        wald_results = ols_model.wald_test_terms()
        
        results[column] = wald_results
        
    return results




# In[8]:


def correlation(df, target_column):
    correlations = []
    p_values = []
    
    for column in df.columns:
        if column != target_column:
            corr, p_val = stats.spearmanr(df[column], df[target_column])
            correlations.append(corr)
            p_values.append(p_val)
    
    results = pd.DataFrame({
        'Correlation': correlations,
        'P-Value': p_values
    }, index=[col for col in df.columns if col != target_column])
    
    return results


# In[9]:


def varclus(df):
    df_var= VarClusHi(df.astype('int'),maxclus=None,maxeigval2=0.7).varclus()
    r= df_var.rsquare
    return r


# In[10]:


def woenum(df,y):
    x= df.select_dtypes(include=['int','float']).columns
    woe_models= {}
    binning_tables= {}
    
    for col in x:
        binning= OptimalBinning(name=col, dtype="numerical", solver="cp")
        binning.fit(df[col],y)
        woe_models[col]= binning
        binning_tables[col]= binning.binning_table.build()
        
    return binning_tables


# In[11]:


def woecat(df,y):
    x= df.select_dtypes(exclude=['int','float']).columns
    woe_models= {}
    binning_tables= {}
    
    for col in x:
        binning= OptimalBinning(name=col, dtype="categorical", solver="mip")
        binning.fit(df[col],y)
        woe_models[col]= binning
        binning_tables[col]= binning.binning_table.build()
        
    return binning_tables


# In[14]:


def woegrafnum(df,y):
    x= df.select_dtypes(include=['int','float']).columns
    woe_models= {}
    binning_tables= {}
    bingrafs= []
    
    for col in x:
        binning= OptimalBinning(name=col, dtype="numerical", solver="cp")
        binning.fit(df[col],y)
        woe_models[col]= binning
        binning_tables[col]= binning.binning_table.build()
        
        bingraf= binning.binning_table.plot(metric='event_rate')
        bingrafs.append(bingraf)
        
    return bingrafs


# In[15]:


def woegrafcat(df,y):
    x= df.select_dtypes(exclude=['int','float']).columns
    woe_models= {}
    binning_tables= {}
    bingrafs= []
    
    for col in x:
        binning= OptimalBinning(name=col, dtype="categorical", solver="mip")
        binning.fit(df[col],y)
        woe_models[col]= binning
        binning_tables[col]= binning.binning_table.build()
        
        bingraf= binning.binning_table.plot(metric='event_rate')
        bingrafs.append(bingraf)
        
    return bingrafs


# In[ ]:




