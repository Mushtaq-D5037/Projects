
"""
@author: Mushtaq Mohammed
"""
import pandas as pd

# read csv file
data = pd.read_csv('wm_new.csv')

# some basic checks
data.columns

# selecting variables
df = data[['age','marital_status','childrens_age','household_size',
           'house_owner','net_worth' ,'children' ,'household_income',
           'occupation' ,'ethnicity' ,'education','state_code',
           # interests
           'interests_travel', 'interests__reading', 'interests__food_cooking',
           'interests_excercise_health', 'interests__movie_music',
           'interests__electronics_computers', 'interests__home_imporvement',
           'interests__investing_finance', 'interests__collectibles_antiques',
           ]]


# missing values
df.isnull().sum()

# Missing value Imputation
df_copy = df.copy()
df_copy.isnull().sum()
df_copy = df_copy.dropna(subset=['occupation'])
df_copy['age'].fillna(round(df_copy['age'].mean(),0),inplace = True)
df_copy['state_code'].fillna(df_copy['state_code'].mode()[0],inplace = True)


# filling childrens_age
df_copy['childrens_age'].dtype
df_copy['childrens_age'] = df_copy['childrens_age'].fillna(9999)
df_copy['childrens_age'] = df_copy['childrens_age'].astype('int') # float to int
df_copy['childrens_age'] = df_copy['childrens_age'].apply(str)    # int   to string
df_copy['childrens_age'] = df_copy['childrens_age'].replace('9999','1000000000000000')

# string len
df_copy['count'] = df_copy['childrens_age'].str.len()
df_copy['count'].value_counts()

def label_age (row):
   # No children
   if ((row['count'] == 16) ):
      return 'No Children' 
   # 00-02
   if ((row['count'] == 15) |
       (row['count'] == 14) |
       (row['count'] == 13) ):
      return '00-02 Age'
   # age 03-05
   if ((row['count'] == 12) |
       (row['count'] == 11) |
       (row['count'] == 10) ):
      return '03-05 Age'

  # age 06-10
   if ((row['count'] == 9) |
       (row['count'] == 8) |
       (row['count'] == 7) ):
      return '06-10 Age'
  # age 06-10
   if ((row['count'] == 6) |
       (row['count'] == 5) |
       (row['count'] == 4) ):
       return '11-15 Age'
   # age 06-10
   if ((row['count'] == 3) |
       (row['count'] == 2) |
       (row['count'] == 1) ):
       return '16-17 Age'
  
df_copy['childrens_age'] = df_copy.apply (lambda row: label_age(row), axis=1)
df_copy['childrens_age'].value_counts()

# data prepartion
df_copy['marital_status'] = df_copy['marital_status'].map({'Married':'Married',
                                                 'inferredMarried':'Married',
                                                 'Single':'Single',
                                                 'inferredSingle':'Single'})

df_copy['occupation'] = df_copy['occupation'].replace({'Self Employed - Professional/Technical':'Professional/Technical',
                                                 'Self Employed - Clerical/White Collar':'Clerical/White Collar',
                                                 'Self Employed - Craftsman/Blue Collar':'Craftsman/Blue Collar',
                                                 'Self Employed - Administration/Managerial':'Administration/Managerial',
                                                 'Self Employed - Homemaker':'Homemaker',
                                                 'Self Employed - Sales/Service':'Sales/Service',
                                                 'Self Employed - Other':'Other',
                                                 'Self Employed - Retired':'Retired'})

# numeric and categoric
import numpy as np
num = df_copy.select_dtypes(include=[np.number])
cat = df_copy.select_dtypes(exclude=[np.number])

num.columns
cat.columns

# scaling numeric columns
from sklearn.preprocessing import StandardScaler

std = StandardScaler()
for i in num:
    df_copy[i] = std.fit_transform(df_copy[i].values.reshape(-1,1))

# taking indexes of categorical column 
cat_columns_index = [ df_copy.columns.get_loc(c) for c in cat.columns if c in df_copy.columns ]


# K-Prototypes
from kmodes.kprototypes import KPrototypes

X = df_copy.values
kproto   = KPrototypes(n_clusters=12)
clusters = kproto.fit_predict(X, categorical = cat_columns_index)

# adding clusters to data
df_copy['cluster'] = clusters
 
# creating segments
seg1  = df_copy[df_copy['cluster']==0].sort_values(['age'],axis=0,ascending=True)
seg2  = df_copy[df_copy['cluster']==1].sort_values(['age'],axis=0,ascending=True)
seg3  = df_copy[df_copy['cluster']==2].sort_values(['age'],axis=0,ascending=True)
seg4  = df_copy[df_copy['cluster']==3].sort_values(['age'],axis=0,ascending=True)
seg5  = df_copy[df_copy['cluster']==4].sort_values(['age'],axis=0,ascending=True)
seg6  = df_copy[df_copy['cluster']==5].sort_values(['age'],axis=0,ascending=True)
seg7  = df_copy[df_copy['cluster']==6].sort_values(['age'],axis=0,ascending=True)
seg8  = df_copy[df_copy['cluster']==7].sort_values(['age'],axis=0,ascending=True)
seg9  = df_copy[df_copy['cluster']==8].sort_values(['age'],axis=0,ascending=True)
seg10 = df_copy[df_copy['cluster']==9].sort_values(['age'],axis=0,ascending=True)
seg11 = df_copy[df_copy['cluster']==10].sort_values(['age'],axis=0,ascending=True)
seg12 = df_copy[df_copy['cluster']==11].sort_values(['age'],axis=0,ascending=True) 
 



