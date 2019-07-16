# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 20:20:23 2019

@author: Shubham
"""

import pandas as pd
import numpy as np
from glob import glob
import logging

def combine_identity_transaction(mode='train'):
    print("Combining {} files".format(mode))
    merged_file_name = mode + '_merged.csv'
    if len(glob(merged_file_name)) >= 1:
        print("Merged file already exists")
        return
    csv_files = glob(mode + "_*.csv")
    csv_read_list = list(map(lambda x: pd.read_csv(x), csv_files))
    combine_csv = csv_read_list[0].merge(csv_read_list[1], on='TransactionID')
    combine_csv.to_csv(merged_file_name, header=True)
    
if __name__ == "__main__":
    modes = ['train', 'test']
    for mode in modes:
        combine_identity_transaction(mode=mode)