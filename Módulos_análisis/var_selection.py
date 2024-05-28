#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score


# In[2]:


def fowardselection(df, y_col, n):
    y = df[y_col]
    x = df.drop(columns=[y_col])
    
    log = LogisticRegression(random_state=0, max_iter=10000)
    sfs = SequentialFeatureSelector(log, k_features=n, forward=True, scoring='roc_auc', cv=5)
    sfs.fit(x, y)
    
    print(sfs.k_feature_names_)


# In[3]:


def backwardselection(df, y_col, n):
    y = df[y_col]
    x = df.drop(columns=[y_col])
    
    log = LogisticRegression(random_state=0, max_iter=10000)
    sfs = SequentialFeatureSelector(log, k_features=n, forward=False, scoring='roc_auc', cv=5)
    sfs.fit(x, y)
    
    print(sfs.k_feature_names_)




# In[ ]:
def log_selection(df, y_col):
    y = df[y_col]
    x = df.drop(columns=[y_col])
    
    log = LogisticRegression(random_state=0, max_iter=10000,penalty='none')
    log.fit(x,y)
    
    print(log.coef_)
    

def randomforest(df, y_col,score):
    y = df[y_col]
    x = df.drop(columns=[y_col])
    
    rf = RandomForestClassifier(random_state=0,criterion=score)
    rf.fit(x,y)
    
    for x, importance in enumerate(rf.feature_importances_):
        print(f"Feature {x}: {importance}")



