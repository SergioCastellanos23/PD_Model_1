#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


# In[2]:


def quantileoutlier(df,column,umbral):
    q1= df[column].quantile(0.25)
    q3= df[column].quantile(0.75)
    IQR= q3-q1
    outliers= df[column][((df[column]<(q1-(umbral)*IQR))|(df[column]>(q3+(umbral)*IQR)))]
    return outliers


# In[3]:


def outlier(df,column):
    q1= df[column].quantile(0.25)
    q3= df[column].quantile(0.75)
    IQR= q3-q1
    outliers= df[column][((df[column]<(q1-1.5*IQR))|(df[column]>(q3+1.5*IQR)))]
    return outliers


# In[4]:


def isolation(df, column, contamination):
    data = df[[column]]
    
    model = IsolationForest(contamination=contamination, random_state=0)
    model.fit(data)
    
    outliers = model.predict(data)
    
    outliers_df = pd.DataFrame({'Outlier': outliers}, index=df.index)
    
    df_outliers = pd.concat([df, outliers_df], axis=1)
    
    df_clean = df_outliers[df_outliers['Outlier'] != -1]
    
    df_clean = df_clean.drop(columns=['Outlier'])
    
    return df_clean


# In[ ]:





# In[ ]:




