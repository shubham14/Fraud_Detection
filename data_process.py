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
import torch
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
        return
    csv_files = glob(mode + "_*.csv")
    csv_read_list = list(map(lambda x: pd.read_csv(x, index_col='TransactionID'),
                                    csv_files))
    combine_csv = csv_read_list[1].merge(csv_read_list[0], how='left', 
                               left_index=True, right_index=True)
    combine_csv.to_csv(merged_file_name, header=True)
    
def processDataFrame(df):
    pass
    

class TransactionDataset(Dataset):
    """
    Fraud Detection datase
    Converting
    """
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.transactions = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.transactions)

    def __getitem__(self, idx):
        y_train = self.transactions['isFraud'].copy()
        # Drop target, fill in NaNs 
        X_train = self.transactions.drop('isFraud', axis=1)
        X_train = X_train.fillna(-999)
        record = X_train.iloc[idx, 0]
        isFraud = self.transactions.iloc[idx, 1:]
        sample = {'record': record, 'isFraud': isFraud}
        return sample
    
if __name__ == "__main__":
    modes = ['train', 'test']
    for mode in modes:
        combine_identity_transaction(mode=mode)
        
    fraud_dataset = TransactionDataset('train_merged.csv')
