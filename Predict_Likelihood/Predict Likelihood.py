#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn
import scipy
import asgl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression, Lasso
from optbinning.scorecard import plot_auc_roc, plot_cap, plot_ks
from scipy import stats
import regressor
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import statsmodels.stats.proportion as proportion
from sklearn.preprocessing import OneHotEncoder
from optbinning import OptimalBinning
from optbinning import OptimalBinningSketch
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import linregress
import pingouin as pg
from varclushi import VarClusHi
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn import linear_model
from yellowbrick.model_selection import RFECV
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from stepwise_regression import step_reg
import xgboost as xgb
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import plot_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.linear_model import Ridge, RidgeCV, RidgeClassifier
from sklearn.linear_model import LassoLarsIC, LassoCV, LassoLarsCV
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import HuberRegressor
from sklearn import ensemble
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgb
from xgboost import cv
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import statsmodels.formula.api as smf
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from scipy.stats import shapiro
import warnings

warnings.filterwarnings('ignore')


# In[2]:


df= pd.read_csv("D:\Documentos\CreditData1.csv")
df


# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()
#All variables are good, none has null data or blanck cells


# In[6]:


df.isna().sum()


# In[7]:


df= df.drop(columns=['OBS#'])
df.reset_index(inplace=True)
df= df.drop(columns=['index'])
df


# In[8]:


df['PRESENT_RESIDENT']= df['PRESENT_RESIDENT'].replace({1:0,2:1,3:2,4:3})
df.rename(columns={'CO-APPLICANT': 'CO_APPLICANT','RADIO/TV': 'RADIO_TV'}, inplace=True)
#df.rename(columns={'CO_APPLICANT': 'GUARANTOR','GUARANTOR': 'CO_APPLICANT'}, inplace=True)


# In[9]:


def type(df,column,types):
    df[column]= df[column].astype(types)
    return


# In[10]:


type(df,'CHK_ACCT',"category")
type(df,'HISTORY', "category")
type(df,'SAV_ACCT',"category")
type(df,'EMPLOYMENT',"category")
type(df,'PRESENT_RESIDENT',"category")
type(df,'JOB',"category")

type(df,'NEW_CAR',"uint8")
type(df,'USED_CAR',"uint8")
type(df,'FURNITURE',"uint8")
type(df,'RADIO_TV',"uint8")
type(df,'EDUCATION',"uint8")
type(df,'RETRAINING',"uint8")
type(df,'MALE_DIV',"uint8")
type(df,'MALE_SINGLE',"uint8")
type(df,'MALE_MAR_or_WID',"uint8")
type(df, 'CO_APPLICANT',"uint8")
type(df,'GUARANTOR',"uint8")
type(df,'REAL_ESTATE',"uint8")
type(df,'PROP_UNKN_NONE',"uint8")
type(df, 'OTHER_INSTALL',"uint8")
type(df,'RENT',"uint8")
type(df,'OWN_RES',"uint8")
type(df, 'TELEPHONE',"uint8")
type(df,'FOREIGN',"uint8")

type(df,'DEFAULT',"int64")


# In[11]:


df.info()


# In[12]:


stat, p_value = shapiro(df)

print(stat, p_value)


# In[13]:


df.hist(figsize=(10,10))


# In[14]:


plt.boxplot(df['AGE'])


# In[15]:


sns.distplot(df['AGE'],bins=7)


# In[16]:


plt.boxplot(df['AMOUNT'])


# In[17]:


sns.distplot(df['AMOUNT'],bins=8)


# In[18]:


plt.boxplot(df['DURATION'])


# In[19]:


sns.distplot(df['DURATION'],bins=6,rug=False)


# #### Podemos concluir que nuestros datos tienen una distribución normal debido a los histogramas antes vistos. Por lo tanto, los evaluaremos como datos robustos

# In[20]:


def outlier(column):
    q1= df[column].quantile(0.25)
    q3= df[column].quantile(0.72)
    IQR= q3-q1
    outliers= df[column][((df[column]<(q1-3.5*IQR))|(df[column]>(q3+3.5*IQR)))]
    return outliers


# In[21]:


outlier('AGE')


# In[22]:


outlier('AMOUNT')


# In[23]:


outlier('DURATION')


# #### Por cuestiones de politicas de credito, eliminaremos los datos en 'AGE' mayores a 69 por considerarlos de alto riesgo.

# In[24]:


df= df.drop(df[df['AMOUNT']>=11998].index)
df.reset_index(inplace=True)
df= df.drop(columns=['index'])
df


# In[25]:


df= df.drop(df[df['DURATION']==72].index)
df.reset_index(inplace=True)
df= df.drop(columns=['index'])
df


# In[26]:


df= df.drop(df[df['AGE']>69].index)
df.reset_index(inplace=True)
df= df.drop(columns=['index'])
df


# In[27]:


df.boxplot(figsize=(8,5), rot=90)


# In[28]:


df['DEFAULT'].value_counts()


# In[29]:


df['DEFAULT'].value_counts(normalize=True)


# In[30]:


sns.countplot(x='DEFAULT', data=df, palette='hls')
plt.show()


# In[31]:


grouped_describe = df.groupby('DEFAULT', axis=0).describe()
grouped_describe['AMOUNT']


# In[32]:


df.corr()


# In[33]:


df.info()


# ## DATA EXPLORATION

# In[34]:


def contingency(column,rot=0):
    y = df['DEFAULT']
    x = df[column]
    table = pd.crosstab(x, y)
    plot = table.div(table.sum(1), axis=0).plot(kind='bar', stacked=True, legend=False, rot=rot)
    return plot


# In[35]:


def graf_func(column):
    column= df[column].astype('int64')
    plot = sns.JointGrid(data=df, x=column)
    plot.plot_joint(sns.histplot)
    plot.plot_marginals(sns.boxplot)
    return plot


# In[36]:


def data_tabla(column):
    grouped_describe = df[column].groupby(df['DEFAULT'], axis=0).describe(include='all')
    return grouped_describe


# In[37]:


def logit(column):
    x= df[column].astype('int')
    y= df['DEFAULT']
    logit= smf.logit('y~x',data=df).fit()
    return (logit.wald_test_terms())


# In[38]:


def logplot1(column, df):
    y = df['DEFAULT']
    x = df.drop(columns='DEFAULT')
    column_data = df[[column]] 
    
    log = LogisticRegression(penalty=None)
    log.fit(X=x, y=y)
    y_pred = log.predict_proba(x)[:, 1]
    
    new = pd.DataFrame(data={'Default': y_pred})
    df3 = pd.concat([column_data, new], axis=1)
    
    plot = sns.lmplot(x=column, y='Default', data=df3)
    return plot


# In[39]:


graf_func('CHK_ACCT')


# In[40]:


contingency('CHK_ACCT')


# In[41]:


data_tabla('CHK_ACCT')


# In[42]:


logit('CHK_ACCT')


# In[43]:


logplot1('DURATION',df)


# In[44]:


graf_func('DURATION')


# In[45]:


contingency('DURATION' )


# In[46]:


data_tabla('DURATION')


# In[47]:


logit('DURATION')


# In[48]:


graf_func('HISTORY')


# In[49]:


contingency('HISTORY')


# In[50]:


data_tabla('HISTORY')


# In[51]:


logit('HISTORY')


# In[52]:


graf_func('NEW_CAR')


# In[53]:


contingency('NEW_CAR')


# In[54]:


data_tabla('NEW_CAR')


# In[55]:


logit('NEW_CAR')


# In[56]:


graf_func('USED_CAR')


# In[57]:


contingency('USED_CAR')


# In[58]:


data_tabla('USED_CAR')


# In[59]:


logit('USED_CAR')


# In[60]:


graf_func('FURNITURE')


# In[61]:


contingency('FURNITURE')


# In[62]:


data_tabla('FURNITURE')


# In[63]:


logit('FURNITURE')


# In[64]:


graf_func('RADIO_TV')


# In[65]:


contingency('RADIO_TV')


# In[66]:


data_tabla('RADIO_TV')


# In[67]:


logit('RADIO_TV')


# In[68]:


graf_func('EDUCATION')


# In[69]:


contingency('EDUCATION')


# In[70]:


data_tabla('EDUCATION')


# In[71]:


logit('EDUCATION')


# In[72]:


graf_func('RETRAINING')


# In[73]:


contingency('RETRAINING')


# In[74]:


data_tabla('RETRAINING')


# In[75]:


logit('RETRAINING')


# In[76]:


logplot1('AMOUNT',df)


# In[77]:


graf_func('AMOUNT')


# In[78]:


data_tabla('AMOUNT')


# In[79]:


logit('AMOUNT')


# In[80]:


contingency('SAV_ACCT')


# In[81]:


graf_func('SAV_ACCT')


# In[82]:


data_tabla('SAV_ACCT')


# In[83]:


logit('SAV_ACCT')


# In[84]:


graf_func('EMPLOYMENT')


# In[85]:


contingency('EMPLOYMENT')


# In[86]:


data_tabla('EMPLOYMENT')


# In[87]:


logit('EMPLOYMENT')


# In[88]:


graf_func('INSTALL_RATE')


# In[89]:


contingency('INSTALL_RATE')


# In[90]:


data_tabla('INSTALL_RATE')


# In[91]:


logit('INSTALL_RATE')


# In[92]:


logplot1('INSTALL_RATE',df)


# In[93]:


graf_func('MALE_DIV')


# In[94]:


contingency('MALE_DIV')


# In[95]:


data_tabla('MALE_DIV')


# In[96]:


logit('MALE_DIV')


# In[97]:


graf_func('MALE_SINGLE')


# In[98]:


contingency('MALE_SINGLE')


# In[99]:


data_tabla('MALE_SINGLE')


# In[100]:


logit('MALE_SINGLE')


# In[101]:


graf_func('MALE_MAR_or_WID')


# In[102]:


contingency('MALE_MAR_or_WID')


# In[103]:


data_tabla('MALE_MAR_or_WID')


# In[104]:


logit('MALE_MAR_or_WID')


# In[105]:


graf_func('CO_APPLICANT')


# In[106]:


contingency('CO_APPLICANT')


# In[107]:


data_tabla('CO_APPLICANT')


# In[108]:


logit('CO_APPLICANT')


# In[109]:


graf_func('GUARANTOR')


# In[110]:


contingency('GUARANTOR')


# In[111]:


data_tabla('GUARANTOR')


# In[112]:


logit('GUARANTOR')


# In[113]:


graf_func('PRESENT_RESIDENT')


# In[114]:


contingency('PRESENT_RESIDENT')


# In[115]:


data_tabla('PRESENT_RESIDENT')


# In[116]:


logit('PRESENT_RESIDENT')


# In[117]:


graf_func('REAL_ESTATE')


# In[118]:


contingency('REAL_ESTATE')


# In[119]:


data_tabla('REAL_ESTATE')


# In[120]:


logit('REAL_ESTATE')


# In[121]:


graf_func('PROP_UNKN_NONE')


# In[122]:


contingency('PROP_UNKN_NONE')


# In[123]:


data_tabla('PROP_UNKN_NONE')


# In[124]:


logit('PROP_UNKN_NONE')


# In[125]:


logplot1('AGE',df)


# In[126]:


logit('AGE')


# In[127]:


graf_func('AGE')


# In[128]:


contingency('AGE')


# In[129]:


data_tabla('AGE')


# In[130]:


graf_func('OTHER_INSTALL')


# In[131]:


contingency('OTHER_INSTALL')


# In[132]:


data_tabla('OTHER_INSTALL')


# In[133]:


logit('OTHER_INSTALL')


# In[134]:


graf_func('RENT')


# In[135]:


contingency('RENT')


# In[136]:


data_tabla('RENT')


# In[137]:


logit('RENT')


# In[138]:


graf_func('OWN_RES')


# In[139]:


contingency('OWN_RES')


# In[140]:


data_tabla('OWN_RES')


# In[141]:


logit('OWN_RES')


# In[142]:


graf_func('NUM_CREDITS')


# In[143]:


contingency('NUM_CREDITS')


# In[144]:


data_tabla('NUM_CREDITS')


# In[145]:


logit('NUM_CREDITS')


# In[146]:


logplot1('NUM_CREDITS',df)


# In[147]:


graf_func('JOB')


# In[148]:


contingency('JOB')


# In[149]:


data_tabla('JOB')


# In[150]:


logit('JOB')


# In[151]:


graf_func('NUM_DEPENDENTS')


# In[152]:


contingency('NUM_DEPENDENTS')


# In[153]:


data_tabla('NUM_DEPENDENTS')


# In[154]:


logit('NUM_DEPENDENTS')


# In[155]:


logplot1('NUM_DEPENDENTS',df)


# In[156]:


graf_func('TELEPHONE')


# In[157]:


contingency('TELEPHONE')


# In[158]:


data_tabla('TELEPHONE')


# In[159]:


logit('TELEPHONE')


# In[160]:


graf_func('FOREIGN')


# In[161]:


contingency('FOREIGN')


# In[162]:


data_tabla('FOREIGN')


# In[163]:


logit('FOREIGN')


# In[164]:


graf_func('DEFAULT')


# In[165]:


contingency('DEFAULT')


# In[166]:


data_tabla('DEFAULT')


# In[167]:


scaler= MinMaxScaler()

x_scaler= scaler.fit_transform(df.drop(columns='DEFAULT'))

x_scaler


# In[168]:


data_scaler= pd.DataFrame(data=(x_scaler), columns=(df.drop(columns='DEFAULT').columns))
df_scal= pd.concat([data_scaler,df['DEFAULT']],axis=1)
df_scal


# In[169]:


sns.kdeplot(data=df_scal)


# In[170]:


correlation,p= stats.spearmanr(df)
correlation=pd.DataFrame(data= correlation, columns= df.columns, index= df.columns)
correlation['DEFAULT']


# In[171]:


pvalue=pd.DataFrame(data= p, columns= df.columns, index= df.columns)
pvalue['DEFAULT']


# In[172]:


correlation= df.corr(method='spearman')
plt.figure(figsize=(25,10))
sns.heatmap(correlation, annot=True, cmap= 'YlGnBu')


# In[173]:


df_var= VarClusHi(df_scal,maxclus=None,maxeigval2=0.7)
df_var.varclus()


# In[174]:


df_var.info


# In[175]:


df_var.rsquare


# In[176]:


corr= stats.spearmanr(df_scal[['DEFAULT','RENT','OWN_RES']])
corr[0][0]


# In[177]:


corr2= stats.spearmanr(df_scal[['DEFAULT','AMOUNT','DURATION']])
corr2[0][0]


# In[178]:


corr3= stats.spearmanr(df_scal[['DEFAULT','HISTORY','NUM_CREDITS']])
corr3[0][0]


# In[179]:


corr4= stats.spearmanr(df_scal[['DEFAULT','NEW_CAR','RADIO_TV']])
corr4[0][0]


# In[180]:


corr5= stats.spearmanr(df_scal[['DEFAULT','TELEPHONE','JOB']])
corr5[0][0]


# In[181]:


df.drop(columns=['DEFAULT']).astype('int64').sum().rank(method='average', ascending=False)


# In[182]:


def woe_iv(column, dtype,user_splits=None):
    x= df[column]
    y= df['DEFAULT']
    #y= default['N_DEFAULT'].values
    optb= OptimalBinning(name='y', dtype=dtype,monotonic_trend='auto',
                         user_splits=user_splits
                         
                        )
    #method : 'uniform', 'quantile', 'cart', 'mdlp'
    #solver : str, optional (default="cp")
    #The optimizer to solve the optimal binning problem. Supported solversare "mip" to choose a mixed-integer programming solver, "cp" to choosea constrained programming solver or "ls" to choose `LocalSolver<https://www.localsolver.com/>`_.
    optb.fit(x,y)
    binning_table= optb.binning_table
    return binning_table.build()


# In[183]:


def woe(column, dtype,bins,method):
    x= df[column]
    y= df['DEFAULT']    #y= default['N_DEFAULT'].values
    optb= OptimalBinning(name='y', dtype=dtype,monotonic_trend='auto',
                         max_n_prebins=bins,prebinning_method= method,
                        )
    #method : 'uniform', 'quantile', 'cart', 'mdlp'
    optb.fit(x,y)
    binning_table= optb.binning_table
    return binning_table.build()


# In[184]:


woe_iv('DURATION','numerical',user_splits=[11,15,24,30])


# In[185]:


woe('AMOUNT','numerical',9,'quantile')


# In[186]:


woe('INSTALL_RATE', 'categorical',4,'uniform')


# In[187]:


woe('AGE', 'numerical',7,'quantile')


# In[188]:


woe('NUM_CREDITS','categorical',4,'uniform')


# In[189]:


woe('NUM_DEPENDENTS', 'categorical',2, 'uniform')


# In[190]:


woe('NEW_CAR', 'categorical',2, 'uniform')


# In[191]:


woe('USED_CAR', 'categorical',2, 'uniform')


# In[192]:


woe('FURNITURE', 'categorical',2, 'uniform')


# In[193]:


woe('RADIO_TV', 'categorical',2, 'uniform')


# In[194]:


woe('EDUCATION', 'categorical',2, 'uniform')


# In[195]:


woe('RETRAINING', 'categorical',2, 'uniform')


# In[196]:


woe('MALE_DIV', 'categorical',2, 'uniform')


# In[197]:


woe('MALE_SINGLE', 'categorical',2, 'uniform')


# In[198]:


woe('MALE_MAR_or_WID', 'categorical',2, 'uniform')


# In[199]:


woe('GUARANTOR', 'categorical',2, 'uniform')


# In[200]:


woe('CO_APPLICANT', 'categorical',2, 'uniform')


# In[201]:


woe('REAL_ESTATE', 'categorical',2, 'uniform')


# In[202]:


woe('PROP_UNKN_NONE', 'categorical',2, 'uniform')


# In[203]:


woe('OTHER_INSTALL', 'categorical',2, 'uniform')


# In[204]:


woe('RENT', 'categorical',2, 'uniform')


# In[205]:


woe('OWN_RES', 'categorical',2, 'uniform')


# In[206]:


woe('TELEPHONE', 'categorical',2, 'uniform')


# In[207]:


woe('FOREIGN', 'categorical',2, 'uniform')


# In[208]:


woe('CHK_ACCT', 'categorical',4, 'uniform')


# In[209]:


woe('HISTORY', 'categorical',5, 'uniform')


# In[210]:


woe('SAV_ACCT', 'categorical',5, 'uniform')


# In[211]:


woe('EMPLOYMENT', 'categorical',5, 'uniform')


# In[212]:


woe('PRESENT_RESIDENT', 'categorical',4, 'uniform')


# In[213]:


woe('JOB', 'categorical',4, 'uniform')


# ## VARIABLE SELECTION

# In[214]:


x=  df.drop(columns=['DEFAULT'])
y= df['DEFAULT']


# In[215]:


kf= StratifiedKFold(n_splits=5)


# #### Forward Selection

# In[216]:


log= LogisticRegression(random_state=0)
sfs= SequentialFeatureSelector(log,k_features=12,forward=True,scoring='roc_auc',cv=kf)
sfs.fit(x,y)

sfs.k_feature_names_


# In[217]:


X1= sm.add_constant(x[['CHK_ACCT',
 'DURATION',
 'HISTORY',
 'USED_CAR',
 'RADIO_TV',
 'EDUCATION',
 'SAV_ACCT',
 'EMPLOYMENT',
 'MALE_SINGLE',
 'CO_APPLICANT',
 'OTHER_INSTALL',
 'FOREIGN']])
Y1=y

logit= sm.Logit(Y1,X1,hasconst=True).fit()
print(logit.summary(),logit.wald_test_terms())


# #### Backward Selection

# In[218]:


log= LogisticRegression(random_state=0)
sfs= SequentialFeatureSelector(log,k_features=12,forward=False,scoring='roc_auc',cv=kf)
sfs.fit(x,y)

sfs.k_feature_names_


# In[219]:


X3= sm.add_constant(x[['CHK_ACCT',
 'DURATION',
 'HISTORY',
 'NEW_CAR',
 'USED_CAR',
 'EDUCATION',
 'SAV_ACCT',
 'EMPLOYMENT',
 'INSTALL_RATE',
 'CO_APPLICANT',
 'OTHER_INSTALL',
 'RENT']])
Y3=y

logit= sm.OLS(Y3,X3,hasconst=True).fit()
print(logit.summary(),logit.wald_test_terms())


# In[220]:


def stepwise_selection(data, target,SL_in=0.05,SL_out =0.1):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<SL_in):
            best_features.append(new_pval.idxmin())
            while(len(best_features)>0):
                best_features_with_constant = sm.add_constant(data[best_features])
                p_values = sm.OLS(target, best_features_with_constant).fit().pvalues[1:]
                max_p_value = p_values.max()
                if(max_p_value >= SL_out):
                    excluded_feature = p_values.idxmax()
                    best_features.remove(excluded_feature)
                else:
                    break 
        else:
            break
    return best_features


# In[221]:


stepwise_selection(x,y)


# In[222]:


X4= sm.add_constant(x[['CHK_ACCT',
 'DURATION',
 'HISTORY',
 'USED_CAR',
 'SAV_ACCT',
 'CO_APPLICANT',
 'NEW_CAR',
 'EDUCATION',
 'OTHER_INSTALL',
 'RENT',
 'INSTALL_RATE',
 'EMPLOYMENT',
 'FOREIGN']])
Y4= y
                   
logit= sm.OLS(Y4,X4).fit()
print(logit.summary(),logit.wald_test_terms())


# Se crearon modelos para medir el rendimiento de las variables y seleccionar las mejores variables, posteriormente se analizarán los modelos

# In[223]:


logistic= LogisticRegression(random_state=0)
logistic.fit(x,y)

param_grid = {
    'penalty': ['none','l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga','elasticnet']
}

grid_search = GridSearchCV(estimator=LogisticRegression(random_state=0), param_grid=param_grid, cv=5)
grid_search.fit(x, y)

print(grid_search.best_params_)
print(grid_search.best_score_)

log1 = LogisticRegression(C= 0.1, solver='liblinear',penalty='l2',random_state=0)
log1.fit(x,y)
print(log1.coef_)


scores= cross_val_score(log1,x,y,cv=kf)
print(scores)
print(scores.mean())


# In[224]:


rf = RandomForestClassifier(random_state=0)
rf.fit(x,y)

print(' ')

param_grid= {'n_estimators': range(1,50),
             'max_depth': range(1,4)}

grid_search = GridSearchCV(rf,param_grid, cv=kf, scoring='accuracy')
grid_search.fit(x, y)
    
print(grid_search.best_params_)
print(grid_search.best_score_)


# In[225]:


rf1 = RandomForestClassifier(random_state=0, max_depth=3, n_estimators=4)
rf1.fit(x,y)

print(' ')

for i, importance in enumerate(rf1.feature_importances_):
    print(f"Feature {i}: {importance}")

print(' ')

scores= cross_val_score(RandomForestClassifier(random_state=0, max_depth= 3, n_estimators= 4),x,y,cv=kf)
print(scores)
print(scores.mean())


# In[226]:


xgb = XGBClassifier(enable_categorical=True,random_state=0)
xgb.fit(x, y)
print(' ')

param_grid = {
    'max_depth': range(1,4),
    'learning_rate': [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],
    #'n_estimators': range(1,50)
}

grid_search = GridSearchCV(xgb, param_grid, cv=kf, scoring='accuracy')
grid_search.fit(x, y)

print(' ')
print(grid_search.best_params_)


# In[227]:


xgb1 = XGBClassifier(enable_categorical=True,random_state=0,learning_rate=0.08,max_depth=3,n_estimators=7)
xgb1.fit(x,y)

print(' ')

for i, importance in enumerate(xgb1.feature_importances_):
    print(f"Feature {i}: {importance}")

print(' ')

scores= cross_val_score(xgb1,x,y,cv=kf)
print(scores)
print(scores.mean())


# In[228]:


gbc= GradientBoostingClassifier(random_state=0,max_features=5)
gbc.fit(x,y)

param_grid = {
    'max_depth': range(1,3),
    'learning_rate': [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],
    'n_estimators': range(1,50)
}

grid_search = GridSearchCV(gbc, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x, y)
print(grid_search.best_params_)


# In[229]:


gbc1 = GradientBoostingClassifier(random_state=0,learning_rate=0.1,max_depth=2,n_estimators=44)
gbc1.fit(x,y)

print(' ')

for i, importance in enumerate(gbc1.feature_importances_):
    print(f"Feature {i}: {importance}")

print(' ')

scores= cross_val_score(gbc1,x,y,cv=kf)
print(scores)
print(scores.mean())


# In[230]:


dt= DecisionTreeClassifier(random_state=0,criterion='log_loss')
dt.fit(x,y)
print(' ')

param_grid = {
    'max_depth': range(1,3)
}

grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x, y)
print(' ')
print(grid_search.best_params_)

dt1= DecisionTreeClassifier(criterion='log_loss', max_depth=2,random_state=0)
dt1.fit(x,y)
print(' ')
for i, importance in enumerate(dt1.feature_importances_):
    print(f"Feature {i}: {importance}")
    
scores= cross_val_score(dt1,x,y,cv=kf)
print(scores)
print(scores.mean())


# In[231]:


abc= AdaBoostClassifier(random_state=0)
abc.fit(x,y)
print(' ')

param_grid = {
    'n_estimators': range(1,50),
    'learning_rate': [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
}

grid_search = GridSearchCV(abc, param_grid, cv=kf, scoring='accuracy')
grid_search.fit(x, y)
print(' ')
print(grid_search.best_params_)


# In[232]:


abc1= AdaBoostClassifier(random_state=0,learning_rate=0.09,n_estimators=100)
abc1.fit(x,y)
print(' ')
for i, importance in enumerate(abc1.feature_importances_):
    print(f"Feature {i}: {importance}")

    
scores= cross_val_score(abc1,x,y,cv=kf)
print(scores)
print(scores.mean())


# In[233]:


lasso = Lasso(random_state=0)
ridge = Ridge(random_state=0)
elasticnet = ElasticNet(random_state=0)

lasso_params = {'alpha': [.01,.1,1,10]}
ridge_params = {'alpha': [.01,.1,1]}
elasticnet_params = {'alpha': [.01,.1,1,10],
                     'l1_ratio': [0.001,0.01,.1,1]}

lasso_grid = GridSearchCV(lasso, lasso_params,cv=kf)
ridge_grid = GridSearchCV(ridge, ridge_params,cv=kf)
elasticnet_grid = GridSearchCV(elasticnet, elasticnet_params,cv=kf)

lasso_grid.fit(x,y)
ridge_grid.fit(x,y)
elasticnet_grid.fit(x,y)

lasso_best_params = lasso_grid.best_params_
ridge_best_params = ridge_grid.best_params_
elasticnet_best_params = elasticnet_grid.best_params_


# In[234]:


lasso_best_params, ridge_best_params, elasticnet_best_params


# In[235]:


lasso = Lasso(alpha=0.01)
ridge = Ridge(alpha=1,tol=0.000001)
elasticnet = ElasticNet(alpha=0.1, l1_ratio=0.1)

lasso.fit(x, y)
ridge.fit(x,y)
elasticnet.fit(x,y)


# In[236]:


lasso.coef_


# In[237]:


ridge.coef_


# In[238]:


elasticnet.coef_


# ## Dividir datos para la selección del mejor modelo

# In[239]:


final_model= df[['DURATION','OTHER_INSTALL','USED_CAR','MALE_SINGLE',
                 'GUARANTOR','OWN_RES','CHK_ACCT','SAV_ACCT','EMPLOYMENT',
                 'FOREIGN']]


# In[240]:


final_model


# In[241]:


x_train,x_test,y_train,y_test= train_test_split(final_model,y,test_size=.30,random_state=0)


# In[242]:


x_train


# In[243]:


param_grid = {
    'penalty': ['none','l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga','elasticnet']
}

grid_search = GridSearchCV(estimator=LogisticRegression(random_state=0), param_grid=param_grid, cv=5)
grid_search.fit(x_train, y_train)

print(grid_search.best_params_)

print(grid_search.best_score_)

y_pred= grid_search.predict(x_test)

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))


# In[244]:


rfc= RandomForestClassifier(random_state=0, max_depth= 3, n_estimators= 4)

rfc.fit(x_train, y_train)

y_pred= rfc.predict(x_test)

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))


# In[245]:


xgb1 = XGBClassifier(enable_categorical=True,random_state=0,learning_rate=0.08,max_depth=3,n_estimators=7)
xgb1.fit(x_train,y_train)

y_pred= xgb1.predict(x_test)

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))


# In[246]:


gbc1 = GradientBoostingClassifier(random_state=0,learning_rate=0.1,max_depth=2,n_estimators=44)

gbc1.fit(x_train,y_train)

y_pred= gbc1.predict(x_test)

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))


# In[247]:


dt1= DecisionTreeClassifier(criterion='log_loss', max_depth=2,random_state=0)
dt1.fit(x_train,y_train)

y_pred= dt1.predict(x_test)

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))


# In[248]:


abc1= AdaBoostClassifier(random_state=0,learning_rate=0.09,n_estimators=100)
abc1.fit(x_train,y_train)

y_pred= abc1.predict(x_test)

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))


# # Creando el SCORECARD

# ## Transform data to simplify 

# In[249]:


x_train['DURATION_11']= np.where(x_train['DURATION']<=11,1,0)

x_train['DURATION_11_15']= np.where((x_train['DURATION']>11) & (x_train['DURATION']<15),1,0)

x_train['DURATION_15_24']= np.where((x_train['DURATION']>15)&(x_train['DURATION']<24),1,0)

x_train['DURATION_24_30']= np.where((x_train['DURATION']>24)&(x_train['DURATION']<30),1,0)

x_train['DURATION_30']= np.where(x_train['DURATION']>30,1,0)


x_test['DURATION_11']= np.where(x_test['DURATION']<=11,1,0)

x_test['DURATION_11_15']= np.where((x_test['DURATION']>11) & (x_test['DURATION']<15),1,0)

x_test['DURATION_15_24']= np.where((x_test['DURATION']>15)&(x_test['DURATION']<24),1,0)

x_test['DURATION_24_30']= np.where((x_test['DURATION']>24)&(x_test['DURATION']<30),1,0)

x_test['DURATION_30']= np.where(x_test['DURATION']>30,1,0)


# In[250]:


x_train['EMPLOYMENT>4']= np.where(x_train['EMPLOYMENT'].isin([3,4]),1,0)
x_test['EMPLOYMENT>4']= np.where(x_test['EMPLOYMENT'].isin([3,4]),1,0)


# In[251]:


x_train= pd.get_dummies(x_train,columns=['CHK_ACCT','SAV_ACCT','USED_CAR','MALE_SINGLE','GUARANTOR','OTHER_INSTALL','OWN_RES','FOREIGN','EMPLOYMENT'])
x_test= pd.get_dummies(x_test,columns=['CHK_ACCT','SAV_ACCT','USED_CAR','MALE_SINGLE','GUARANTOR','OTHER_INSTALL','OWN_RES','FOREIGN','EMPLOYMENT'])


# In[252]:


x_train= x_train.drop(columns=['DURATION'])
x_test= x_test.drop(columns=['DURATION'])


# In[253]:


x_train['EMPLOYMENT>4'].value_counts()


# In[254]:


x_final= sm.add_constant(x_train['CHK_ACCT_0'])
y_final= y_train
logit= sm.OLS(y_final,x_final).fit()
print(logit.summary2(),logit.wald_test_terms(), np.exp(logit.params))


# In[255]:


x_final= sm.add_constant(x_train['CHK_ACCT_1'])
y_final= y_train
logit= sm.OLS(y_final,x_final).fit()
print(logit.summary2(),logit.wald_test_terms(), np.exp(logit.params))


# In[256]:


x_final= sm.add_constant(x_train['CHK_ACCT_2'])
y_final= y_train
logit= sm.OLS(y_final,x_final).fit()
print(logit.summary2(),logit.wald_test_terms(), np.exp(logit.params))


# In[257]:


x_final= sm.add_constant(x_train['CHK_ACCT_3'])
y_final= y_train
logit= sm.OLS(y_final,x_final).fit()
print(logit.summary2(),logit.wald_test_terms(), np.exp(logit.params))


# In[258]:


x_final= sm.add_constant(x_train[['USED_CAR_0']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[259]:


x_final= sm.add_constant(x_train[['USED_CAR_1']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[260]:


x_final= sm.add_constant(x_train[['SAV_ACCT_0']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[261]:


x_final= sm.add_constant(x_train[['SAV_ACCT_1']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[262]:


x_final= sm.add_constant(x_train[['SAV_ACCT_2']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[263]:


x_final= sm.add_constant(x_train[['SAV_ACCT_3']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[264]:


x_final= sm.add_constant(x_train[['SAV_ACCT_4']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[265]:


x_final= sm.add_constant(x_train[['EMPLOYMENT_0']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[266]:


x_final= sm.add_constant(x_train[['EMPLOYMENT_1']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[267]:


x_final= sm.add_constant(x_train[['EMPLOYMENT_2']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[268]:


x_final= sm.add_constant(x_train[['EMPLOYMENT_3']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[269]:


x_final= sm.add_constant(x_train[['EMPLOYMENT_4']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[270]:


x_final= sm.add_constant(x_train[['EMPLOYMENT>4']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[271]:


x_final= sm.add_constant(x_train[['MALE_SINGLE_0']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[272]:


x_final= sm.add_constant(x_train[['MALE_SINGLE_1']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[273]:


x_final= sm.add_constant(x_train[['GUARANTOR_0']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[274]:


x_final= sm.add_constant(x_train[['GUARANTOR_1']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[275]:


x_final= sm.add_constant(x_train[['OTHER_INSTALL_0']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(),
      np.exp(logit.params))


# In[276]:


x_final= sm.add_constant(x_train[['OTHER_INSTALL_1']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(),
      np.exp(logit.params))


# In[277]:


x_final= sm.add_constant(x_train[['FOREIGN_0']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[278]:


x_final= sm.add_constant(x_train[['FOREIGN_1']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[279]:


x_final= sm.add_constant(x_train[['OWN_RES_0']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[280]:


x_final= sm.add_constant(x_train[['OWN_RES_1']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[281]:


x_final= sm.add_constant(x_train[['DURATION_11']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[282]:


x_final= sm.add_constant(x_train[['DURATION_11_15']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[283]:


x_final= sm.add_constant(x_train[['DURATION_15_24']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[284]:


x_final= sm.add_constant(x_train[['DURATION_24_30']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[285]:


x_final= sm.add_constant(x_train[['DURATION_30']])
y_final= y_train
logit= sm.Logit(y_final,x_final).fit()
print(logit.summary2(),
      logit.wald_test_terms(), 
      np.exp(logit.params))


# In[286]:


x_train= x_train.drop(columns=['EMPLOYMENT_0','EMPLOYMENT_1','EMPLOYMENT_2','EMPLOYMENT_3','EMPLOYMENT_4'],axis=0)
x_test= x_test.drop(columns=['EMPLOYMENT_0','EMPLOYMENT_1','EMPLOYMENT_2','EMPLOYMENT_3','EMPLOYMENT_4'],axis=0)


# In[287]:


x_train


# In[290]:


final_model1= x_train[['CHK_ACCT_0','CHK_ACCT_1','CHK_ACCT_2',
                            'DURATION_11','DURATION_11_15','DURATION_15_24',
                            'USED_CAR_1','SAV_ACCT_0','EMPLOYMENT>4',
                            'MALE_SINGLE_1','GUARANTOR_1','OTHER_INSTALL_1',
                            'FOREIGN_1','OWN_RES_1']]

x_newtest= x_test[['CHK_ACCT_0','CHK_ACCT_1','CHK_ACCT_2',
                            'DURATION_11','DURATION_11_15','DURATION_15_24',
                            'USED_CAR_1','SAV_ACCT_0','EMPLOYMENT>4',
                            'MALE_SINGLE_1','GUARANTOR_1','OTHER_INSTALL_1',
                            'FOREIGN_1','OWN_RES_1']]


# In[291]:


log1 = LogisticRegression(C= 0.1, solver='liblinear',penalty='l2',random_state=0)
log1.fit(final_model1,y_train)
coefc= np.array([log1.coef_])

for i, coef in enumerate(log1.coef_[0]):
    print(f"Coeficiente {i}: {coef}")


# In[292]:


odds_ratio= np.exp(log1.coef_)
odds_ratio.tolist()


# In[293]:


cov_matrix = np.linalg.inv(np.dot(final_model1.T, final_model1))

std_errors = np.sqrt(np.diagonal(cov_matrix))

print("Errores estándar de los coeficientes:")
for i, coef in enumerate(log1.coef_[0]):
    print(f"Coeficiente {i}: {coef} +/- {std_errors[i]}")


# In[294]:


c = np.array(log1.coef_).flatten()

new_applicant = np.array([1,0,0,0,0,0,0,1,0,0,0,1,0,0])

probabilidad = 1 / (1 + np.exp(-(np.dot(new_applicant, c))))

print(probabilidad)


# In[295]:


from optbinning import BinningProcess
from optbinning import Scorecard


# In[296]:


binning_process= BinningProcess(final_model1.columns.values)
binning_process


# In[297]:


log1 = LogisticRegression(C= 0.1, solver='liblinear',penalty='l2',random_state=0)
scaling_method= "pdo_odds"
scaling_method_data= {"pdo": 20, "odds": 30, "scorecard_points": 600}

scorecard= Scorecard(binning_process=binning_process,estimator= log1, scaling_method= scaling_method, scaling_method_params= scaling_method_data, intercept_based=False, reverse_scorecard=True,)
scorecard.fit(final_model1, y_train)


# In[298]:


scorecard_summary= scorecard.table('detailed').round(2)
scorecard_summary


# In[299]:


x_newtest.loc[:,"score"]= scorecard.score(x_newtest)

plot_ks(y_test, x_newtest.score)


# In[300]:


plot_auc_roc(y_test, x_newtest.score)


# In[301]:


plot_cap(y_test, x_newtest.score)


# In[302]:


x_newtest


# In[ ]:




