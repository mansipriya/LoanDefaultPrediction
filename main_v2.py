# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 19:53:56 2019

@author: VE00YM015
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Loading files as dataframe
train = pd.read_csv('train.csv', )
test = pd.read_csv('test.csv')

#Feature Engineering
date = []
for i in test['origination_date']:
    date.append(i.split('/')[1])
test['origination month'] = date
    
date = []
for i in train['origination_date']:
    date.append(i.split('/')[1])
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
    m = i.split('-')[0]
    if(m == "Feb"):
        date.append(2)
    elif(m == "Mar"):
        date.append(3)
    elif(m == "Apr"):
        date.append(4)
    elif(m == "May"):
        date.append(5)
train['first_payment_month'] = date

test['origination month'] = pd.to_numeric(test['origination month'])
train['origination month'] = pd.to_numeric(train['origination month'])
test['first_payment_month'] = pd.to_numeric(test['first_payment_month'])
train['first_payment_month'] = pd.to_numeric(train['first_payment_month'])

test['difference'] = test['origination month'] - test['first_payment_month']
train['difference'] = train['origination month'] - train['first_payment_month']

df = pd.concat([train, test], axis=0, sort=True)

df['total'] = df['unpaid_principal_bal'] * 100 / df['loan_to_value']
df['amount_to_pay'] = ((1 + df['interest_rate']/100)**(df['loan_term']/12))*df['total']
df['monthly_installment'] = df['amount_to_pay']/(df['loan_term']/12)
df['income'] = df['total'] * 100/df['debt_to_income_ratio']
df['sum_of_m'] = df['m1'] + df['m2'] + df['m3'] + df['m4'] + df['m5'] + df['m6'] + df['m7'] + df['m8'] + df['m9'] + df['m10'] + df['m11'] + df['m12']
df['average_score'] = (df['borrower_credit_score'] + df['co-borrower_credit_score'])/df['number_of_borrowers']

count = []
for index, row in df.iterrows(): 
    if(row['m12'] > 0):
        c = 0
        count.append(c)
        continue
    if(row['m11'] > 0):
        c = 1
        count.append(c)
        continue
    if(row['m10'] > 0):
        c = 2
        count.append(c)
        continue
    if(row['m9'] > 0):
        c = 3
        count.append(c)
        continue
    if(row['m8'] > 0):
        c = 4
        count.append(c)
        continue
    if(row['m7'] > 0):
        c = 5
        count.append(c)
        continue
    if(row['m6'] > 0):
        c = 6
        count.append(c)
        continue
    if(row['m5'] > 0):
        c = 7
        count.append(c)
        continue
    if(row['m4'] > 0):
        c = 8
        count.append(c)
        continue
    if(row['m3'] > 0):
        c = 9
        count.append(c)
        continue
    if(row['m2'] > 0):
        c = 10
        count.append(c)
        continue
    if(row['m1'] > 0):
        c = 11
        count.append(c)
        continue
    else:
        c = 12
        count.append(c)
df['no_payment_count'] = count

#df['paid_amount'] = df['amount_to_pay'] - df['unpaid_principal_bal']
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

continuous = ['no_payment_count','average_score','borrower_credit_score','co-borrower_credit_score','debt_to_income_ratio','difference','first_payment_month','insurance_percent','interest_rate','loan_term','loan_to_value','m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12','number_of_borrowers','origination month','unpaid_principal_bal','total','amount_to_pay','monthly_installment','income']

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
clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)

# Train the classifier
clf.fit(X_train, Y_train)
feat_labels = X_train.columns
# Print the name and gini importance of each feature
for feature in zip(feat_labels, clf.feature_importances_):
    print(feature)
 
#After analyzing feature importance we remove following columns
#X_train.drop(['m9','m10','m11','number_of_borrowers','origination month','total','amount_to_pay'], axis=1, inplace=True)
#X_test.drop(['m9','m10','m11','number_of_borrowers','origination month','total','amount_to_pay'], axis=1, inplace=True)

X_train.drop(['m1','m2','m3','m4','m5','m6','number_of_borrowers','difference','insurance_percent','sum_of_m'], axis=1, inplace=True)
X_test.drop(['m1','m2','m3','m4','m5','m6','number_of_borrowers','difference','insurance_percent','sum_of_m'], axis=1, inplace=True)
  
from scipy.stats import chi2_contingency
csq=chi2_contingency(pd.crosstab(X_train['financial_institution'], Y_train))
print("P-value: ",csq[1])

csq=chi2_contingency(pd.crosstab(X_train['insurance_type'], Y_train))
print("P-value: ",csq[1])

csq=chi2_contingency(pd.crosstab(train['source'], Y_train))
print("P-value: ",csq[1])

csq=chi2_contingency(pd.crosstab(train['loan_purpose'], Y_train))
print("P-value: ",csq[1])

X_train.drop(['financial_institution','insurance_type','loan_purpose_A23','loan_purpose_B12','loan_purpose_C86','source_X','source_Y','source_Z'], axis=1, inplace=True)
X_test.drop(['financial_institution','insurance_type','loan_purpose_A23','loan_purpose_B12','loan_purpose_C86','source_X','source_Y','source_Z'], axis=1, inplace=True)

d = X_train.corr()
d.to_csv("corr.csv")

X_train.drop(['m9','m10','total','borrower_credit_score'], axis=1, inplace=True)
X_test.drop(['m9','m10','total','borrower_credit_score'], axis=1, inplace=True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3) 
rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [700,1000,850,1100],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [7,8,9],
    'criterion' :['gini']
}
scoring = {'AUC': 'roc_auc'}
from sklearn.model_selection import GridSearchCV
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 3,verbose = 5,scoring = 'roc_auc',refit = 'AUC',n_jobs = -1)
CV_rfc.fit(X_train, Y_train)
CV_rfc.best_params_

rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 1000, max_depth=8, criterion='gini')
rfc1.fit(X_train, Y_train)
pred=rfc1.predict(x_test)
from sklearn.metrics import accuracy_score
print("Accuracy for Random Forest on CV data: ",accuracy_score(y_test,pred))
pred_final = rfc1.predict(X_test)

op=pd.DataFrame(test['loan_id'])
op['m13']=pred_final

op.to_csv("submission_v13.csv")

 