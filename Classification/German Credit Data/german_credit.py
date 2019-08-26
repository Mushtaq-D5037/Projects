"""
@author: Mushtaq Mohammed
"""
# import libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# read csv file
df = pd.read_csv('german_credit_data.csv')

# common checks
df.shape
df.columns
df.info()

# display top 10 rows
df.head(10)

# checking distribution of response variable
df['Risk'].value_counts().plot(kind='pie',autopct='%.f%%',explode=(0.1,0),shadow=True,startangle=90)
# observations
# 30% bad 
# 70% good 
# and also data in not imbalanced


# =============================================================================
# Data Analysis and visualizations
# =============================================================================

# statistic details of each variable
stats = df.describe().T

# histogram
df.hist(bins = 110)


# boxplot
df.boxplot()
# credit amount has high number of outliers

# bivariate analysis
def count_groupBar(dataframe,target,column,title=None,xlabel=None,ylabel=None):
    ax = sns.countplot(dataframe[column],hue=dataframe[target])
    
    # if want to add title and axis labels
    title  = title
    xlabel = xlabel
    ylabel = ylabel
    
    plt.title ( title  )
    plt.xlabel( xlabel )
    plt.ylabel( ylabel )
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2,height+5,'{:0.0f}'.format(height),ha='center')
    plt.show()



def defaultRate(df,response_variable,independent_variable,cr_tab_column,title=None,xlabel=None,ylabel=None):
    '''
    function to check the likelihood of accepting or rejecting
    1.create a crosstab dataframe,calculate % of accepting or rejecting
    2.plot graph
    '''
    crosstab_df = pd.crosstab(df[response_variable],df[independent_variable]).apply(lambda x:x/x.sum()*100)
    crosstab_df = crosstab_df.T
    
    ax = crosstab_df[cr_tab_column].sort_values(ascending=True).plot(kind='barh')
    
    title  = title
    xlabel = xlabel
    ylabel = ylabel
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    for p,label in zip(ax.patches,crosstab_df[cr_tab_column].sort_values(ascending = True).round(1).astype(str)):
        ax.text(p.get_width()+0.8,
                p.get_y()+ p.get_height()-0.5, 
                label+'%', 
                ha = 'center', 
                va='bottom')

count_groupBar(df,'Risk','Purpose','Total Count of Credit Purpose','purpose of credit','count')
defaultRate(df,'Risk','Purpose','bad''Default Rate of Credit Purpose','risk','Purpose')


count_groupBar(df,'Risk','Duration')
defaultRate(df,'Risk','Duration','bad')

# =============================================================================
# Model Building
# =============================================================================
# missing values
df.isnull().sum()

# filling missing categoric values with mode
df['Saving accounts'].fillna(df['Saving accounts'].mode()[0],inplace =True)
df['Checking account'].fillna(df['Checking account'].mode()[0],inplace = True)

# numeric and categoric
num = df.select_dtypes(include =[np.number])
cat = df.select_dtypes(exclude =[np.number])

for i in cat.columns:
    print(df[i].value_counts(),'\n')

# converting categoric columns to numeric columns using label Encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

mappings = []

for col in cat.columns:
    # to know the mappings of the LabelEncoder
    cat[col]  = le.fit(cat[col])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    mappings.append(le_name_mapping)
    
    # LabelEncoding
    cat[col] = le.fit_transform(df[col])
    

# one hot encoding and dropping first variable
# to avoid correlation between the variables (dummy variable trap)
df = pd.get_dummies(df,columns = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose','Job'],drop_first=True)


# deleteing unncessary columns
df.drop('Unnamed: 0',axis = 1 , inplace =True)
    
# correlation matrix
corr = df.corr()
sns.heatmap(corr,annot = True)
print(corr['Risk'].sort_values)
# credit amount and duration are positively correlated

# train_test_split
from sklearn.model_selection import train_test_split,cross_val_score

X = df.drop(['Risk'],axis = 1)
y = df['Risk']

# scaling
#from sklearn.preprocessing import StandardScaler

#stdScaler = StandardScaler()
#X['Duration'] = stdScaler.fit_transform(X[['Duration']].values)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 42)

# model selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# spot check algorithm
classifiers = {'LR' :LogisticRegression(),
               'SVC':SVC(),
               'DT' :DecisionTreeClassifier(),
               'RF' :RandomForestClassifier()
              }


for name,model in classifiers.items():
    accuracy = cross_val_score(estimator = model, X=X_train, y=y_train, cv=5)
    print(name,':',accuracy.mean())

# Logistic Regression    
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# accuracy_score and confusion_matrix
from sklearn.metrics import accuracy_score ,confusion_matrix,classification_report

print('Accuracy',accuracy_score(y_test,y_pred))
print('ConfusioMatrix:','\n',confusion_matrix(y_test,y_pred))
print('Classification Report:\n',classification_report(y_test,y_pred))


# tuning GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = {
              'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] 
             }

grid = GridSearchCV(model, param_grid, cv=10, scoring = 'accuracy')
grid.fit(X_train, y_train)
y_pred_gridSearch = grid.predict(X_test)
print('Accuracy',accuracy_score(y_test,y_pred_gridSearch))
print('ConfusioMatrix:','\n',confusion_matrix(y_test,y_pred_gridSearch))
print('Classification Report:\n',classification_report(y_test,y_pred_gridSearch))


