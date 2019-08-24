# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:15:32 2019

@author: VE00YM015
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Loading files as dataframe
train = pd.read_csv('train.csv', )
test = pd.read_csv('test.csv')

def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)

             
display_all(df.describe(include='all').T)
df.isna().sum()

#Feature Engineering
date = []
for i in test['origination_date']:
    date.append(i.split('/')[1])
test['origination month'] = date
    
date = []
for i in train['origination_date']:
    date.append(i.split('-')[1])
train['origination month'] = date

date = []
for i in test['first_payment_date']:
    m = i.split('-')[0]
    if(m == "Feb"):
        date.append(2)
    elif(m == "Mar"):
        date.append(3)
    elif(m == "Apr"):
        date.append(4)
    elif(m == "May"):
        date.append(5)
test['first_payment_month'] = date


date = []
for i in train['first_payment_date']:
       date.append(i.split('/')[0])
train['first_payment_month'] = date

test['origination month'] = pd.to_numeric(test['origination month'])
train['origination month'] = pd.to_numeric(train['origination month'])
test['first_payment_month'] = pd.to_numeric(test['first_payment_month'])
train['first_payment_month'] = pd.to_numeric(train['first_payment_month'])

test['difference'] = test['origination month'] - test['first_payment_month']
train['difference'] = train['origination month'] - train['first_payment_month']

df = pd.concat([train, test], axis=0, sort=True)

#Label Encoding
# convert to cateogry dtype
df['financial_institution'] = df['financial_institution'].astype('category')
# convert to category codes
df['financial_institution'] = df['financial_institution'].cat.codes

# subset all categorical variables which need to be encoded
categorical = ['source','loan_purpose']

for var in categorical:
    df = pd.concat([df, 
                    pd.get_dummies(df[var], prefix=var)], axis=1)
    del df[var]

# drop the variables we won't be using
df.drop(['loan_id','origination_date','first_payment_date'], axis=1, inplace=True)

continuous = ['borrower_credit_score','co-borrower_credit_score','debt_to_income_ratio','difference','first_payment_month','insurance_percent','interest_rate','loan_term','loan_to_value','m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12','number_of_borrowers','origination month','unpaid_principal_bal']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

for var in continuous:
    df[var] = df[var].astype('float64')
    df[var] = scaler.fit_transform(df[var].values.reshape(-1, 1))

#Getting test and train
X_train = df[pd.notnull(df['m13'])].drop(['m13'], axis=1)
Y_train = df[pd.notnull(df['m13'])]['m13']
X_test = df[pd.isnull(df['m13'])].drop(['m13'], axis=1)

#Feature Importance
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the classifier
clf.fit(X_train, y_train)
feat_labels = X_train.columns
# Print the name and gini importance of each feature
for feature in zip(feat_labels, clf.feature_importances_):
    print(feature)

d = X_train.corr()
d.to_csv("corr.csv")

#After analyzing corr we remove following columns
X_train.drop(['m9','m10','m11','number_of_borrowers','origination month'], axis=1, inplace=True)
X_test.drop(['m9','m10','m11','number_of_borrowers','origination month'], axis=1, inplace=True)

from scipy.stats import chi2_contingency
csq=chi2_contingency(pd.crosstab(X_train['financial_institution'], Y_train))
print("P-value: ",csq[1])

from scipy.stats import chi2_contingency
csq=chi2_contingency(pd.crosstab(X_train['insurance_type'], Y_train))
print("P-value: ",csq[1])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3) 
rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [200, 500,1000],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
from sklearn.model_selection import GridSearchCV
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(x_train, y_train)
CV_rfc.best_params_

rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 500, max_depth=8, criterion='entropy')
rfc1.fit(x_train, y_train)
pred=rfc1.predict(x_test)
from sklearn.metrics import accuracy_score
print("Accuracy for Random Forest on CV data: ",accuracy_score(y_test,pred))
pred_final = rfc1.predict(X_test)

op=pd.DataFrame(test['loan_id'])
op['m13']=pred_final

op.to_csv("submission.csv")