# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 20:20:23 2019

@author: Shubham
"""

'''
0 count = 132915, 1 count = 11318, clearly unbalanced
'''

import pandas as pd
import numpy as np
from glob import glob
import logging
from sklearn import preprocessing, metrics
from torch.utils.data import Dataset, DataLoader
import warnings
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

def combine_identity_transaction(mode='train'):
    print("Combining {} files".format(mode))
    merged_file_name = mode + '_merged.csv'
    print("File name is {}".format(merged_file_name))
    if len(glob(merged_file_name)) >= 1:
        print("Merged file already exists")
        return pd.read_csv(merged_file_name)
    csv_files = glob(mode + "_*.csv")
    csv_read_list = list(map(lambda x: pd.read_csv(x, index_col='TransactionID'),
                                    csv_files))
    combine_csv = csv_read_list[1].merge(csv_read_list[0], how='left', 
                               left_index=True, right_index=True)
    combine_csv.to_csv(merged_file_name, header=True)
    return pd.read_csv(merged_file_name)
    
def processDataFrame(train, test):
    y_train = train['isFraud'].copy()
    # Drop target, fill in NaNs 
    X_train = train.drop('isFraud', axis=1)
    X_test = test.copy()
    X_train = X_train.fillna(-999)
    X_test = X_test.fillna(-999)
    for f in X_train.columns:
        if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(X_train[f].values) + list(X_test[f].values))
            X_train[f] = lbl.transform(list(X_train[f].values))
            X_test[f] = lbl.transform(list(X_test[f].values)) 
    return X_train, y_train, X_test
    

class TransactionDataset(Dataset):
    """
    Fraud Detection Pytorch dataset
    """
    def __init__(self, X_train, y_train):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        record = self.X_train.values[idx, :]
        isFraud = self.y_train[idx]
        sample = {'record': record, 'isFraud': isFraud}
        return sample
    
if __name__ == "__main__":
    modes = ['train', 'test']
    l = []
    for mode in modes:
        x = combine_identity_transaction(mode=mode)
        l.append(x)
    X_train, y_train, X_test = processDataFrame(l[0], l[1])
    fraud_dataset = TransactionDataset(X_train, y_train)
