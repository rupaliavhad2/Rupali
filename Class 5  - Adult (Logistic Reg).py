# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd

###ADULT DATASET
#%%
#IMPORT DATA SET
adult_df=pd.read_csv(r'C:\Users\DELL\Desktop\RUPALI\dataset py\adult_data.csv',
                  header = None, delimiter=' *, *', engine='python')
#delimiter=' *, *' it is used for special characteristics used in place of missing values
adult_df.head()
#%%
#TO GET ALL COLUMN NAMES
pd.set_option('display.max_columns',None)
#pd.set_option('display.max_rows',None)
#%%
#TO CHECK THE SHAPE OF DATA
adult_df.shape
#%%
#TO NAME THE COLUMNS
adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'capital_gain', 'capital_loss',
'hours_per_week', 'native_country', 'income']
adult_df.head()
#%%
###PREPROCESSING THE DATA
#%%
#FINDING MISSING VALUES
adult_df.isnull().sum()
#since missing values are denoted by a character ,it is not registered as missing value
#to convert the character to Nan values and find missing values
adult_df=adult_df.replace(['?'],np.nan)
adult_df.isnull().sum()
#%%
#CREATE A COPY OF THE DATAFRAME
adult_df_rev = pd.DataFrame.copy(adult_df)
#adult_df_rev.describe(include='all')
#%%
##DROPPING SOME COLUMNS
adult_df_rev=adult_df_rev.drop(["education","fnlwgt"],axis=1)
adult_df_rev.head()
#%%
#TREATMENT OF MISSING VALUES
#replace the missing values with values in the top row of each column
for value in ['workclass','occupation','native_country']:
    adult_df_rev[value].fillna(adult_df_rev[value].mode()[0],inplace=True)
adult_df_rev.workclass.mode()
adult_df_rev.isnull().sum()
#adult_df_rev.head()
#%%
#FOR TREATMENT OF MISSING VALUES IN MULTIPLE COLUMNS
"""
for x in adult_df_rev.columns[:]:
    if adult_df_rev[x].dtype=='object':
        adult_df_rev[x].fillna(adult_df_rev[x].mode()[0],inplace=True)
    elif adult_df_rev[x].dtype=='int64'| adult_df_rev[x].dtype=='float64':
        adult_df_rev[x].fillna(adult_df_rev[x].mean(),inplace=True)
"""
#%%
adult_df_rev.workclass.value_counts()
#%%
#TO CONVERT CATEGORICAL DATA TO LEVELS
#create a list of all categorical variables
colname=['workclass','marital_status','occupation','relationship',
         'race','sex','native_country','income']
colname
#to convert the data to levels
#FOR PREPROCESSING THE DATA
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for x in colname: adult_df_rev[x]=le.fit_transform(adult_df_rev[x])
adult_df_rev.head()
#0==> <=50k
#1==> >50k
#%%
#CHECK THE DATA TYPE
adult_df_rev.dtypes
#%%
#TO CONVERT DATA FRAME TO ARRAYS
X= adult_df_rev.values[:,:-1]
Y= adult_df_rev.values[:,-1]
#%%
