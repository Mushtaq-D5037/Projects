# importing libraries
import pandas as pd
import numpy  as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

# loading data
data = '/../creditcard.csv'
df = pd.read_csv(data)

# descriptive stats
stats = df.describe()
df.info()
df.shape

# numeric and categoric column
num_col = df.select_dtypes(include = [np.number]).columns.tolist()
cat_col = df.select_dtypes(exclude = [np.number]).columns.tolist()

# visualizing target variable distribution
df['Class'].value_counts().plot(kind   = 'bar', 
                                title  = 'Target variable Distribution\nNon-Fraud:0 | Fraud:1',
                                ylabel = 'Count',
                                xlabel = 'Class',
                                rot    = 0)
# checking target variable distribution
df['Class'].value_counts(normalize =True)
# Observation
# Highly Imbalanced data


# splitting into train and test
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state = 42,  stratify = y)
# Stratify: ensures same proportion of samples to be present in y_train and y_test
y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)

# =============================================================================
# Sampling
# =============================================================================
# sampling only Training data
# sampling strategies
# Random Under Sampling : reduces majority class to match minority class
# Random Over Sampling  : increases minority class to match majority class
# SMOTE                 : increases minority class to match majority class by creating synthetic samples
# and many more

# Random Under Sampling
rus = RandomUnderSampler(sampling_strategy=0.3)
X_rus, y_rus= rus.fit_resample(X_train, y_train)
print(f'sampled trained data percentage:\n{y_rus.value_counts(normalize =True)}')
print(f'sampled trained data count:\n{y_rus.value_counts()}')

# =============================================================================
# Feature Selection
# =============================================================================
# Quick Shortlisting variables Strategy
# 1. Removing constant variables : standar deviation = 0
# 2. Removing Quasi constant variables
# 3. Removing columns with High precentage of missing value
# 4. Removing Highly Correlated Variables
# 5. Removing Low Univariate ROC-AUC curve ( cut-off - 50% or 55%)

# 1. Constant Features
# variables having only one value
constant_features = [ col for col in X_rus.columns if X_rus[col].std()==0]
# -----------------------------------------------------------------------------

# 2. Quasi Constant Features
# variables with 99% of only one level 
# for example, 0 are 99% and 1 are less than 1%
# 0 --> 0.999971
# 1 --> 0.000029
from sklearn.feature_selection import VarianceThreshold

feature_selector = VarianceThreshold(threshold =0.01) 
feature_selector.fit(X_rus)
non_quasi_constant = X_rus.columns[feature_selector.get_support()]
sum(feature_selector.get_support())
Quasi_constant_features = [ c for c in X_rus.columns if c not in non_quasi_constant ]
# -----------------------------------------------------------------------------

# 3. correlated variable
import seaborn as sns
corr_matrix = X_rus.corr()
sns.heatmap(corr_matrix, annot=True)

def corr(data, threshold=None):
    ''' function to check threshold value w.r.t
    each row (i) and column(j) of correlation matrix'''
    
    corr_col = set()  # to avoid duplicated columns
    corr_mat = data.corr()
    for i in range(len(corr_mat.columns)):
        for j in range(i):
            if(abs(corr_mat.iloc[i,j] > threshold)):
                colname = corr_mat.columns[i]
                corr_col.add(colname)
    return corr_col

correlated_columns = corr(data = X_rus, threshold = 0.8)
print(f'number of correlated features: {len(correlated_columns)}')
# X_rus.drop(correlated_columns, axis=1, inplace=True)   
# X_test.drop(correlated_columns, axis=1, inplace=True)   

# correlated features Dataframe
corr_matrx = X_rus.corr()
corr_matrx = corr_matrx.abs().unstack()
corr_matrx.sort_values(ascending = False, inplace =True)
# taking features with correlation above threshold > 0.8
corr_matrx = corr_matrx[corr_matrx > 0.8 ]
corr_matrx = corr_matrx[corr_matrx < 1]
corr_matrx = pd.DataFrame(corr_matrx).reset_index()
corr_matrx.columns = ['Feature1', 'Feature2', 'Correlation']

# creating correlated feature groups
grouped_features = []
corr_groups = []

for f in corr_matrx.Feature1.unique():
    if f not in grouped_features:
        # Find all features correlated to a single feature
        correlated_block = corr_matrx[corr_matrx['Feature1'] == f]
        grouped_features = grouped_features + list(correlated_block['Feature2'].unique()) + [f]
        
        # Append block of features to the list
        corr_groups.append(correlated_block)
print(f'found {len(corr_groups)} correlated feature groups')
print(f'out of {X_rus.shape[1]} total features.')
# Investigating groups
# Now to decide which variable to keep and which one to remove
# check each group individually
# remove variable with high percentage of missing value 
# if above is not the case, build a random forest model on each group and 
# check variable importance
# keep variable with high importance and remove others
# or
# if algorithm is not sensitive to correlated features, remove correlated features at the end
# but make sure after removing correlated features, model accuracy should be same
for group in corr_groups:
    print(group)

# group = corr_groups[2]
for g_idx in range(len(corr_groups)):
    group = corr_groups[g_idx]
    print('\n')
    print(f"Group:{group['Feature1'].unique()}")
    for f in group['Feature2'].unique():
        print(X_rus[f].isnull().sum())
# Observation 
# No missing values

# build RandomForest model on each group and check importance 
from sklearn.ensemble import RandomForestClassifier

features = list(group['Feature2'].unique()) + list(group['Feature1'].unique())
rfc = RandomForestClassifier(n_estimators=20, random_state=101, max_depth=4)
rfc.fit(X_rus[features].fillna(0), y_train)   
# Get Feature Importance using RFC
importance = pd.concat([pd.Series(features), pd.Series(rfc.feature_importances_)], axis=1)
importance.columns = ['feature', 'importance']
importance.sort_values(by='importance', ascending=False)
# -----------------------------------------------------------------------------

# 4.Univariate ROC-AUC Score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

roc_score = []
clf = DecisionTreeClassifier()
for feature in X_rus.columns:
    clf.fit(X_rus[feature].fillna(0).to_frame(), y_rus)    
    #to_frame(): because we are using only one feature, it becomes series
    #it expects a dataframe, so converting to dataframe
    y_score = clf.predict_proba(X_test[feature].fillna(0).to_frame())
    roc_score.append(roc_auc_score(y_test, y_score[:, 1]))
 
# let add variable names
roc_values = pd.Series(roc_score)   
roc_values.index = X_rus.columns
roc_values = roc_values.sort_values(ascending = False) 
roc_values.sort_values(ascending = False).plot.bar() 

# Selecting Threshold
# any score with 50% means it is a random model
# meaning it is able to predict 50% True and 50% times False
# so Threshold can be 50% or 55% or more
# Selecting Threshold = 55%
len(roc_values[roc_values > 0.55])
drop_variables = list(roc_values[roc_values < 0.55].index)
drop_variables = drop_variables + ['Time']

# Final Variables
X_rus  = X_rus.drop(drop_variables, axis = 1)
X_test = X_test.drop(drop_variables, axis = 1)
print(f'Shape of train data: {X_rus.shape}')
print(f'Shape of test  data: {X_test.shape}')

# =============================================================================
# Selecting best suitable model
# =============================================================================
# K-Fold cross validation can be used to spot check which model is best suitable
# there is no former rule to seleck 'K' value
# in general K=5 or K= 10 is used
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
# evaluate each model in turn

results = []
names = []
model_result_dict = {}
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_rus, y_rus, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    model_result_dict[name] = cv_results.mean()
 
# Observation
# Logistic Regression and Random Forest showing great accuracy
# Let us build these models

# =============================================================================
#  Model Building
# =============================================================================
# 1.Logistic Regression
lr = LogisticRegression()
lr_train= lr.fit(X_rus, y_rus)
lr_pred = lr.predict(X_test)
lr_pred_train = lr.predict(X_rus)

# 2.RandomForest
rf = RandomForestClassifier()
rf_train = rf.fit(X_rus, y_rus)
rf_pred = rf.predict(X_test)
rf_pred_train = rf.predict(X_rus)

# Random Forest Feature Importance
feature  =  list(X_rus.columns)  
feature_imp  = pd.Series(rf.feature_importances_, feature).sort_values(ascending = False)
feature_imp.plot(kind = 'barh')


# =============================================================================
# # Model Evaluation
# =============================================================================
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc

# Checking Overfitting or Underfitting
print('Logistic Regression Trainig Accuracy - {}'.format(accuracy_score(lr_pred_train,y_rus)))
print('Logistic Regression Testing Accuracy - {}'.format(accuracy_score(lr_pred,y_test)))

# Logistic Regression
lr_accuracy= accuracy_score(y_test, lr_pred)
lr_cnfmtrx = confusion_matrix(y_test, lr_pred)
lr_clsrprt = classification_report(y_test, lr_pred)
lr_roc = roc_auc_score(y_test, lr_pred)

print(f'LogisticRegression: Test Accuracy - {lr_accuracy}')
print(f'LogisticRegression: ROC_AUC - {lr_roc}')
print(f'LogisticRegression: Confusion Matrix -\n{lr_cnfmtrx}')
print(f'LogisticRegression: Classification Report -\n{lr_clsrprt}')
# Observation
# Recall and ROC_AUC is poor

# RandomForest
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_cnfmtrx  = confusion_matrix(y_test, rf_pred)
rf_clsrprt  = classification_report(y_test, rf_pred)
rf_roc = roc_auc_score(y_test, rf_pred)

print(f'RandomForest: Test Accuracy - {rf_accuracy}')
print(f'RandomForest: ROC_AUC - {rf_roc}')
print(f'RandomForest: Confusion Matrix -\n{rf_cnfmtrx}')
print(f'RandomForest: Classification Report -\n{rf_clsrprt}')
# Observation
# Random Forest outperforming

# plotting roc curve
fpr,tpr,thresholds = roc_curve(rf_pred,y_test)
auc_score = auc(fpr,tpr)
print('AUC-Score : %0.2f%%'%(auc_score * 100))

#Plotting ROC-Curve
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr,tpr,label = 'Random Forest (auc = %0.2f%%)'%(auc_score *100))
plt.plot([0,1],[0,1],'k--')
plt.legend(loc = 'lower right')
plt.title('ROC-Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# =============================================================================
# Model Tuning
# =============================================================================
# Tuning Parameters using GridSearchCV
from sklearn.model_selection import GridSearchCV

#preparing params grid for Random Forest
parameters = {'bootstrap': [True],
              'max_depth': [80, 90, 100, 110],
              'max_features': [2, 3],
              'min_samples_leaf': [3, 4, 5],
              'min_samples_split': [8, 10, 12],
              'n_estimators': [100, 200, 300, 1000]
             }

grid_search = GridSearchCV(estimator = rf,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

result = grid_search.fit(X_train,y_train)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# prediction
grid_search_predict = grid_search.predict (X_test)
print(confusion_matrix(y_test,grid_search_predict))       # confusion Matrix
print(classification_report(y_test,grid_search_predict))  # classification report
