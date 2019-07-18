# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 20:20:23 2019

@author: Shubham
"""

'''
0 count = 132915, 1 count = 11318, clearly overbalanced
'''

import pandas as pd
import numpy as np
from glob import glob
import logging

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
    combine_csv = csv_read_list[0].merge(csv_read_list[1], how='left', 
                               left_index=True, right_index=True)
    combine_csv.to_csv(merged_file_name, header=True)
    
if __name__ == "__main__":
    modes = ['train', 'test']
    for mode in modes:
        combine_identity_transaction(mode=mode)
