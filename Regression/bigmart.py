# -*- coding: utf-8 -*-
"""
@author: Mushtaq Mohammed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# hypothesis
# item price > 0
# item sales > 0

# read csv files
train = pd.read_csv("store_Train.csv")
test =  pd.read_csv("store_Test.csv")

# combine training and testing data into one and after data cleaning again split it
train['type'] = 'train'
test ['type'] = 'test'

# concatenating
df = pd.concat([train,test],axis = 0) 

# some basic checks
df.head()
df.columns
df.info()

# statistical analysis
statistics = df.describe()

# histogram
df.hist()

# missing values
df.isnull().sum()

# Categoric and Numeric columns
cat = df.select_dtypes(exclude = [np.number])
num = df.select_dtypes(include = [np.number])

# checking all the categorical_data
for col in cat:
    print('\n',col)
    print(df[col].value_counts())
# observations 
# in Item_Fat_Content Low Fat has a typo as LF and low fat  and also Regular as reg

# replacing the typo errors in Item_fat_content
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF'     :'Low Fat',
                                                         'reg'    :'Regular',
                                                         'low fat':'Low Fat'})
# Feature Engineering
# Creating a Column[Item_Type_ID] based on first two letter of [Item_Identifier]
df['Item_Type_ID'] = [letters[0:2] for letters in df['Item_Identifier']]
df['Item_Type_ID'] = df['Item_Type_ID'].map({'FD':'Food',
                                             'DR':'Drink',
                                             'NC':'Non-Consumable'})           # renaming them with more understandable names

df[df['Item_Type_ID']=='Non-Consumable']['Item_Fat_Content']
# observation
# Low Fat --> type id is Non-consumable ( which makes no sense)

# changing the value of a column based on another column  
# so  renaming mapping of non-consumable from low fat to non-edible 
df['Item_Fat_Content'][df['Item_Type_ID']=='Non-Consumable'] = 'Non-Edible'
df['Item_Fat_Content'].value_counts()


# dropping item visibility with 0
df = df[~(df['Item_Visibility']==0)].reset_index(drop =True)


# finding missing values
df.isnull().sum()
# observation 
# Item_Outlet_Sales (Dependent variable)
# Item_Weight
# Outlet_size  has missing data

    
# Handling Missing Values
df[df['Item_Weight'].isna()]['Item_Type_ID'].value_counts()   
# observation
# Item_Weight nan values are food drink and non-consumables
# so replacing Item_Weights with its respective mean values of food drink and non-consumables
# when replacing with mean check for outliers
# coz outliers effects the mean values

# Box-plot for outlier detection
df.boxplot(['Item_Weight'])

# first filling mean with 0 and then replacing it with its respective mean values
df['Item_Weight'] = df['Item_Weight'].fillna(0)
# cross checking
df[df['Item_Weight']==0]['Item_Weight'].value_counts()
df['Item_Type_ID'][df['Item_Weight']== 0 ].value_counts()


# calculating Mean of Item_Visibility with respect to Each Item_Type
Food  = df['Item_Weight'][df['Item_Type_ID']=='Food'] 
Drink = df['Item_Weight'][df['Item_Type_ID']=='Drink'] 
NC    = df['Item_Weight'][df['Item_Type_ID']=='Non-Consumable'] 

FMean = Food.mean()
DMean = Drink.mean()
NCMean= NC.mean()
 
# replacing 0 with its corresponding Mean
Food.replace(0,FMean,inplace = True)
Drink.replace(0,DMean,inplace = True)
NC.replace(0,NCMean,inplace = True)

df['Modified_Item_Weight'] = pd.concat([Food,Drink,NC],axis=0,ignore_index=False)

# Outlet_Size filling missing values
df['Outlet_Size'].value_counts()
df['Outlet_Size'].fillna('Unknown',inplace = True)


# creating one more column of outlet years
df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']
df['Outlet_Years'].value_counts()

# cat columns
cat = df.select_dtypes(exclude = [np.number])

# one Hot Encoding Categorical columns
# 1.Label Encode
# 2.apply pd.dummies or OneHotEncoding()
from sklearn.preprocessing import LabelEncoder

lableEncoder = LabelEncoder()
for i in cat:
    if i not in ['type']:
        df[i] = lableEncoder.fit_transform(df[i])
    

# Variance Inflation Factor to remove collinearity between the variables
# threshold > 10 is considered a high collinearity,threshold = 5 as Medium  (a thumsup rule)  
from statsmodels.stats.outliers_influence import variance_inflation_factor

independent_variables = [col for col in df.columns if col not in ['Item_Outlet_Sales','Item_Identifier',
                                                                  'Item_Weight','Outlet_Establishment_Year',
                                                                  'type']]
X_vif = df[independent_variables]

thresh = 10

for i in np.arange(0,len(independent_variables)):
    vif = [variance_inflation_factor(X_vif[independent_variables].values, ix) for ix in range(X_vif[independent_variables].shape[1])]
    maxloc = vif.index(max(vif))
    if (max(vif) > thresh):
        print ("vif :", vif)
        print('dropping:\n' + X_vif[independent_variables].columns[maxloc] + ' at index: ' + str(maxloc),'\n')
        del independent_variables[maxloc]
    else:
        break
    

new_df = pd.concat([X_vif,df[['Item_Outlet_Sales','type']]],axis = 1)


#Divide into test and train:
new_train = df[df['type']=='train']
new_test =  df[df['type']=='test']

#Drop unnecessary columns:
new_train = new_train.drop(['type'],axis=1)
new_test  = new_test.drop(['Item_Outlet_Sales','type'],axis=1)


#Define target and ID columns:
from sklearn import metrics


predictors = [x for x in new_train.columns if x not in ['Item_Outlet_Sales']]

X = new_train[predictors]
y = new_train['Item_Outlet_Sales']

# Random Forest Model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=200,
                              max_depth=5     , 
                              min_samples_leaf=100,
                              n_jobs=4)
model.fit(X,y)
y_predict = model.predict(new_train[predictors])

print('RMSE:',round(np.sqrt(metrics.mean_squared_error(new_train['Item_Outlet_Sales'],y_predict)),2))
    

coef = pd.Series(model.feature_importances_, predictors).sort_values(ascending=False)
coef.plot(kind='barh', title='Feature Importances')
