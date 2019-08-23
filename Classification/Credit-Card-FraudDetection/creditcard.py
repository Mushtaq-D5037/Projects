# -*- coding: utf-8 -*-

#
# 1.Prepare Problem [Load libraries & load data]
#

# loading libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('creditcard.csv')       # loading data

#
# 2.Summarizing the data [1.Discriptive Statistics & 2.Data Visualization]
#

# Discriptive Statistics

df.shape              # gives total number of rows and columns
df.head()             # gives top 5 rows of the dataframe
                      # identify 'features' and 'labels/target'

df.describe()         # gives statistical analysis - like mean mode median count standard diviation min max 

#
# visualize data
#

# Pandas: BarPlot visualization
# rot = 0 (rotates the x-axis labels horizontally) 
ax = df['Class'].value_counts().plot(kind = 'bar',title = 'Class Distribution\n 0:NoFrauds || 1:Frauds',rot = 0)
ax.set_xticklabels(df['Class'].value_counts())

# printing the percentage of the Non-Frauds and Frauds
# round()- rounding off to two Decimal points using round function
print (' 0:Non-Fraud\n 1:Frauds','\n',
      round( df['Class'].value_counts(normalize = True) * 100, 2)
      )

#
# 3.Prepare Data [1.DataCleaninng 2.Feature Selection & 3.DataTransform]
#

# 1.DataCleaning
# Fill Missing Values
# Convert Categorical Columns to Numeric Format

df.isnull().sum()  # Finding Missing Values in each Columns
# 0 - represents NO Missing Values
# other than 0 represents number of missing values

# 2.Feature Selection
# Finding Correlation between the output and features
# Finding correlation between the features using Variance Inflation Factor

# ***But Before doing this we need To Resample the data ***
# Resampling
# If we don't resample the data,
# we will encounter 'overfitting problem', and Wrong Co-relation
# as the data is imbalanced our classifiction model will assume the Frauds as Non-Frauds

# SMOTE
# Here we Resample our Data using 'SMOTE' Technique

from sklearn.model_selection import train_test_split

# Before Splitting the data taking all the features in X variable and label in y variable
# here i am dropping Time and Amount columns as i don't find it to be useful
# axis=1 means "column"
X = df.drop(['Time','Amount','Class'],axis = 1)
y = df['Class']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

# Right Way of Resampling
# 1.split the 'Original Train data ' into train & test
# 2.oveSample or underSample the splitted train data

# Step 1
X_train_smote,X_test_smote,y_train_smote,y_test_smote = train_test_split(X_train,y_train,test_size = 0.2,random_state =42)

# step 2
from imblearn.over_sampling import SMOTE
smote = SMOTE()

X_smote,y_smote = smote.fit_sample(X_train_smote,y_train_smote)

# converting the X_smote and y_smote into Dataframe to Visualize easily in pandas
df_smote = pd.DataFrame(X_smote)
df_smote.columns = X.columns   #renaming the column names with the Original ones'v1 - v28'
df_smote['target'] = y_smote

df_smote['target'].value_counts().plot(kind = 'bar',title ='SMOTE Class Distribution\n 1:Frauds|0:No-Frauds',rot = 0)
print( round ( df_smote['target'].value_counts(normalize = True)*100,2))

# finding Correlation 
corr_smote =df_smote.corr()
# visualizing corrleation using seaborn
fig,ax = plt.subplots(figsize = (25,15))
sns.heatmap(corr_smote,annot =True,ax=ax)

corr_smote['target'].sort_values(ascending=False)
# positive corelated - v4 , v11,v2
# negative corelated - v14,v12,v10,v16,v9,v3,v17


# box plotting
# Highly Positive & Negatively Correlated variables 
 
correlated_features = ['V4','V11','V2','V10','V16','V14','V12','V9','V3']

for i in range(len(correlated_features)):
    figure = plt.figure()
    ax = sns.boxplot(x='target', y=correlated_features[i], data=df_smote)


def remove_outlier(df, col):
    
    """ removing Oultiers using interquartile range method"""
    
    q1 = df[col].quantile(0.25)     # 1st Quartile
    q3 = df[col].quantile(0.75)     # 3rd Quartile
    iqr = q3-q1                     # Interquartile range
    lower_limit  = q1 - (1.5*iqr)   # lower limit cutoff      
    upper_limit  = q3 + (1.5*iqr)   # upper limit cutoff
    df = df[~( (df < lower_limit) | ( df > upper_limit)).any(axis=1)]
    return df

df_smote = remove_outlier(df_smote, correlated_features )

#
# 4.Evaluate Algorithms
#

# model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

classifiers = {'LR':LogisticRegression(),
               'RF':RandomForestClassifier(),
               'SVC':SVC()
               }

# k-fold cross validation
from sklearn.model_selection import cross_val_score

for name,model in classifiers.items():
    accuracy = cross_val_score(estimator = model,X=X_smote,y=y_smote,cv=10,n_jobs = -1)
    print(accuracy.mean())
    
model = RandomForestClassifier(random_state = 42)   # initializing classifier
model.fit(X_smote,y_smote)         # fit the model on sampled data
prediction=model.predict(X_test)   # perform predictions on Original test set

# Tuning Parameters using GridSearchCV
from sklearn.model_selection import GridSearchCV

#preparing params grid
parameters = {'bootstrap': [True],
              'max_depth': [80, 90, 100, 110],
              'max_features': [2, 3],
              'min_samples_leaf': [3, 4, 5],
              'min_samples_split': [8, 10, 12],
              'n_estimators': [100, 200, 300, 1000]
             }

grid_search = GridSearchCV(estimator = model,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)

grid_search = grid_search.fit(X_smote,y_smote)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


# metrics
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,prediction))       # confusion Matrix
print(classification_report(y_test,prediction))  # classification report

# auc score and roc curve
from sklearn.metrics import roc_curve,auc

rfc_fpr,rfc_tpr,_ = roc_curve(y_test,model.predict(X_test))
rfc_auc = auc(rfc_fpr,rfc_tpr)
print('RandomForestClassifier-auc : %0.2f%%'%(rfc_auc * 100))

#roc curve
plt.figure()
plt.plot(rfc_fpr,rfc_tpr,label ='RFC(auc = %0.2f%%)'%(rfc_auc *100))
plt.plot([0,1],[0,1],'k--')
plt.legend(loc = 'lower right')
plt.title('Smote with RandomForestClassifier\nROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
