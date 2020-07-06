
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# reading data
df = pd.read_csv('creditcard.csv')

# basic checks
df.head()
df.shape
df.columns

# descriptive statistics
df.describe()

# checkig target class distribution
ax = df['Class'].value_counts().plot(kind ='bar',title = 'Class Distribution \n 0:NoFrauds || 1:Frauds',rot = 0)
ax.grid()
total = len(df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2, height + 2 ,'{0:.1%}'.format(height/total),ha = 'center')
    plt.show()
    
    
# checking null values
df.isnull().sum()

# checking correlation
corr = df.corr()
sns.heatmap(corr, annot = True)  


# boxplot
sns.boxplot(y = 'Amount',data = df)

def remove_outlier(df, col):
    
    """ removing Oultiers using interquartile range method"""
    
    q1 = df[col].quantile(0.25)     # 1st Quartile
    q3 = df[col].quantile(0.75)     # 3rd Quartile
    iqr = q3-q1                     # Interquartile range
    lower_limit  = q1 - (1.5*iqr)   # lower limit cutoff      
    upper_limit  = q3 + (1.5*iqr)   # upper limit cutoff
    df = df[~( (df < lower_limit) | ( df > upper_limit)).any(axis=1)]
    return df

df = remove_outlier(df, 'Amount' )


# Multicolinearity check VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df.drop(['Class'],axis = 1)
y = df['Class']


vif = pd.DataFrame()    
vif['vif_factor']  =[ variance_inflation_factor (X.values,i) for i in range(X.shape[1])]
vif['feature']  = X.columns


# Balancing data
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(sampling_strategy = 0.1,random_state = 42)
X_rus,y_rus = rus.fit_sample(X,y)

# value counts
y_rus.value_counts()

# visualizing sampled data
ax = y_rus.value_counts().plot(kind ='bar',
                               title='Resampled Class Distribution \n 0:NoFrauds || 1:Frauds',
                               rot  = 0)
total = len(y_rus)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2, height+2,'{0:.1%}'.format(height/total),ha ='center')
    plt.show()
    
    

# Scaling data
from sklearn.preprocessing import StandardScaler

stdScaler = StandardScaler() 
X_scaled = stdScaler.fit_transform(X_rus)   

# Training and Testing
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_rus, y_rus,test_size = 0.2, random_state = 42)


# Model Building
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 42)
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

# Training accuracy
y_pred_train = lr.predict(X_train)


# model validation metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc

# Checking Overfitting or Underfitting
print('Trainig Accuracy - {}'.format(accuracy_score(y_pred_train,y_train)))
print('Testing Accuracy - {}'.format(accuracy_score(y_pred,y_test)))

# metric validation
print('confusion matrix \n {}'.format(confusion_matrix(y_pred,y_test)))
print('classification Report \n {}'.format(classification_report(y_pred,y_test)))

# roc curve
fpr,tpr,thresholds = roc_curve(y_pred,y_test)
auc_score = auc(fpr,tpr)
print('AUC-Score : %0.2f%%'%(auc_score * 100))

#Plotting ROC-Curve
plt.figure()
plt.plot(fpr,tpr,label = 'Logistic Regression (auc = %0.2f%%)'%(auc_score *100))
plt.plot([0,1],[0,1],'k--')
plt.legend(loc = 'lower right')
plt.title('ROC-Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# k-fold cross validation
from sklearn.model_selection import cross_val_score

accuracy = cross_val_score(estimator = lr, X = X_train, y= y_train, cv = 5, n_jobs = -1)
print('Cross Val Score Accuracy {}'.format(accuracy.mean()))

from sklearn.ensemble import RandomForestClassifier

model  = RandomForestClassifier(random_state = 42)
model.fit(X_train,y_train)
y_pred_rf = model.predict(X_test)

# Feature Importance
feature  =  list(X.columns)  
feature_imp  = pd.Series(model.feature_importances_, feature).sort_values(ascending = False)
feature_imp.plot(kind = 'barh')



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
                           cv = 10)

grid_search.fit(X_train,y_train)
grid_search_predict = grid_search.predict (X_test)

print(confusion_matrix(y_test,grid_search_predict))       # confusion Matrix
print(classification_report(y_test,grid_search_predict))  # classification report
