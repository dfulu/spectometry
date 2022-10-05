import numpy as np
import pandas as pd
import xarray as xr
import dask

import matplotlib.pyplot as plt


from tqdm import tqdm
import time
import os
import pickle

import spect
import nb10

import imp
import copy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("n", help="which model number 1-6",
                    type=int)
args = parser.parse_args()
n = args.n
print(n)
assert n in np.arange(1,7)

results_dir = 'nb12v2_results'

def make_submission(ds_pred, message, folderpath):
    df = ds_pred.to_dataframe().unstack()
    df.columns = df.columns.droplevel(None)
    df.columns.name=None
    os.makedirs(f"{results_dir}/{folderpath}")
    df.to_csv(f"{results_dir}/{folderpath}/submission.csv")
    with open(f"{results_dir}/{folderpath}/note.txt", 'w') as f:
        f.write(f"{folderpath}: {message}")
    os.system(f"zip -r {results_dir}/{folderpath}.zip {results_dir}/{folderpath}")
    os.system(f"rm -rf {results_dir}/{folderpath}")
    
ds_bin = spect.decode_bins(
    dask.compute(
        xr.open_zarr('../data/temp_binned_1.zarr'), 
        scheduler='threads'
    )[0]
)

ds_final = ds_bin.where(ds_bin.split!='test', drop=True)

seeds = [987, 4532, 98111, 6542, 14454, 454]


boosted = nb10.XG2CVsplit(patience=10)
boosted.fit(ds_final, folds=5, kfold_random_seed=seeds[n-1], same_params_all_columns=True, n_trials=100)
boosted.drop_studies()
with open(f'{results_dir}/nb12-model-cv-{n}.pkl','wb') as f:
    pickle.dump(boosted, f)
