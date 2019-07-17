# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 22:16:12 2019

@author: Shubham
"""

import pandas as pd
from sklearn import preprocessing
import xgboost as xgb

def train(X_train, y_train, X_test):
    # Label Encoding
    for f in X_train.columns:
        if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(X_train[f].values) + list(X_test[f].values))
            X_train[f] = lbl.transform(list(X_train[f].values))
            X_test[f] = lbl.transform(list(X_test[f].values)) 
    
    
    print("Start training classfier")
    clf = xgb.XGBClassifier(n_estimators=500,
                            n_jobs=4,
                            max_depth=9,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            missing=-999)
    
    clf.fit(X_train, y_train)
    print("Ended classifier training")
    return X_test, clf

def infer(clf, X_test):
    sample_submission = pd.read_csv('sample_submission.csv', 
                                    index_col='TransactionID')
    sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]
    sample_submission.to_csv('simple_xgboost.csv')