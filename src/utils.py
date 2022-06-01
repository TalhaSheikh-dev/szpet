import datetime
import os
import sys
import re
import random
import subprocess
from shutil import copytree, ignore_patterns
import numpy as np
import pandas as pd
import torch

global device; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_dir(dir_name):
    '''
    Makes a directory if it doesn't exists yet
    Args:
        dir_name: directory name
    '''
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def make_exp_dir(base_exp_dir):
    '''
    Makes an experiment directory with timestamp
    Args:
        base_output_dir_name: base output directory name
    Returns:
        exp_dir_name: experiment directory name
    '''
    now = datetime.datetime.now()
    ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                            now.second)
    exp_dir_name = os.path.join(base_exp_dir, ts)
    make_dir(exp_dir_name)

    src_file = os.path.join(exp_dir_name, 'src')

    if 'SZ_ROOT' not in os.environ:
        os.environ['SZ_ROOT'] = '/content'

    copytree(os.path.join(os.environ['SZ_ROOT'], "src"), src_file,  ignore=ignore_patterns('*.pyc', 'tmp*'))

    return exp_dir_name


def print_mem_usage(loc):
    '''
    Print memory usage in GB
    :return:
    '''
    print("%s mem usage: %.3f GB, %.3f GB, %.3f GB" % (loc, float(torch.cuda.memory_allocated() / 1e9), float(torch.cuda.memory_reserved() / 1e9),  float(torch.cuda.max_memory_allocated() / 1e9)))
    sys.stdout.flush()


def update_dict_val_store(dict_val_store, dict_update_val, grad_accumulation_factor):
    '''
    Update dict_val_store with dict_update_val
    :param dict_val_store:
    :param dict_update_val:
    :return:
    '''
    if dict_val_store is None:
        dict_val_store = dict_update_val
    else:
        for k in dict_val_store.keys():
            dict_val_store[k] += dict_update_val[k] / grad_accumulation_factor

    return dict_val_store

def get_avg_dict_val_store(dict_val_store, num_batches=100):
    '''
    Get average dictionary val
    :param dict_val_store:
    :param eval_every:
    :return:
    '''
    dict_avg_val = {}

    for k in dict_val_store.keys():
        dict_avg_val[k] = float('%.3f' % (dict_val_store[k].detach().cpu().item() / num_batches))

    return dict_avg_val


def dataframe_to_jsonl(df, fn):
    with open(fn, 'w') as f:
        f.write(df.to_json(orient='records', lines=True))


def format_df_for_pet_pattern(df, text_columns, label_column, exclude=['Unlabeled']):
    df = df.copy() # prevent overwriting
    
    # Filter companies that are unlabeled & aren't in the label exclusion list
    df = df.loc[(~df[label_column].isna()) & (~df[label_column].isin(exclude))]
    
    # Assign labels & index to df that pet pattern is expecting
    df['LBL'] = df[label_column]
    df['idx'] = df.index

    # Get all the text columns
    filter_cols = ['idx', 'LBL']
    for idx, col in enumerate(text_columns):
        text_col = 'TEXT{}'.format(idx+1)
        df[text_col] = df[col]
        filter_cols.append(text_col)
    
    # Print out the mapping between the original columns and the new
    original_cols = ['df.index', label_column]
    original_cols.extend(text_columns)
    print('Original column : Pattern column')
    print('---------------------------------')
    for original_col, filter_col in zip(original_cols, filter_cols):
        print('{} : {}'.format(original_col, filter_col))
    print('---------------------------------')

    return df[filter_cols].copy()


def batch_data_from_dataframe(df, desc_cols=['Company Description'], batch_size=2):
    df = df.copy()
    text_cols = []
    for idx, desc_col in enumerate(desc_cols):
        col = 'TEXT{}'.format(idx+1)
        df[col] = df[desc_col].values
        text_cols.append(col)
    
    records = df[text_cols].to_dict(orient='records')
    batches = []

    records_len = len(records)
    for i in range(0, records_len, batch_size):
        batch = records[i:min(i + batch_size, records_len)]
        batch = {'input': {k: [r[k] for r in batch] for k in batch[0].keys()}}
        batches.append(batch)

    return batches