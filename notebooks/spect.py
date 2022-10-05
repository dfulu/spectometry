import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from glob import glob
from tqdm import tqdm
import numba as nb
import dask

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle
from skmultilearn.model_selection import iterative_train_test_split

from typing import Optional, List, Union
import warnings


def base2_split(indices, y, splits):
    log2_splits = np.log2(splits)
    assert log2_splits==int(log2_splits)
    if log2_splits==1:
        indices_1, _, indices_2, _ = iterative_train_test_split(indices, y, test_size=0.5)
        return [indices_1, indices_2]
    else:
        indices_1, y_1, indices_2, y_2 = iterative_train_test_split(indices, y, test_size=0.5)
        return base2_split(indices_1, y_1, splits//2)+base2_split(indices_2, y_2, splits//2)

def cross_val_splits(ds, splits, random_seed=None, pairs=True):
    np.random.seed(random_seed)
    y = ds.labels.values
    indices = shuffle(ds.sample_id.values[:,None])
    partitions = base2_split(indices, y, splits=splits)
    ds_train_test = []
    for p in partitions:
        if pairs:
            x = (ds.drop_sel(sample_id=p[:,0]), ds.sel(sample_id=p[:,0]))
        else:
            x = ds.sel(sample_id=p[:,0])
        ds_train_test.append(x)
    return ds_train_test


BINS = np.arange(0, 1100, 50)

@nb.jit(nopython=True)
def array_to_square_array(x, col):
    col_values = np.unique(x[:,col])
    new_array = [[[0.] for j in range(x.shape[1])] for i in col_values]
    col_counts = [0 for i in col_values]
    max_count = 0
    
    for i in range(x.shape[0]):
        row = x[i]
        
        col_index = np.argwhere(col_values==row[col]).item()
        col_counts[col_index] += 1
        
        for n in range(max_count - col_counts[col_index]):
            for j in range(len(row)):
                new_array[col_index][j].append(np.nan)
            col_counts[col_index] += 1
        
        for j in range(len(row)):
            new_array[col_index][j].append(row[j])

        max_count = max(max_count, col_counts[col_index])
    
    for col_index in range(len(col_values)):
        for n in range(max_count - col_counts[col_index]):
            for j in range(len(row)):
                new_array[col_index][j].append(np.nan)
    return np.array(new_array)[...,1:], col_values

def reshape_df_to_ds(df):
    ds_sample_parts = []
    
    x, mz = array_to_square_array(df.values, 2)
    
    ts = np.nanmean(x[:,0,:], axis=0)
    temps = np.nanmean(x[:,1,:], axis=0)
    abunds = x[:,3,:]
    index = np.arange(x.shape[-1])
    
    ds_new = xr.DataArray(abunds, dims=['mz', 'index'], coords=[mz, index], name='abundance').to_dataset()

    ds_new['time'] = xr.DataArray(ts, dims=['index'], coords=[ds_new.index.values], name='time')
    ds_new['temp'] = xr.DataArray(temps, dims=['index'], coords=[ds_new.index.values], name='temp')
    return ds_new


@nb.jit(nopython=True)
def round_discard(x, fill=-1, tol=.2):
    l = [0]
    for v in x:
        if ((v<int(v)-v)>tol) and (int(v+tol)!=(int(v)+1)):
            l.append(int(fill))
        else:
            l.append(int(v+tol))
    return np.array(l)[1:]

def drop_frac_mz_with_rounding(ds, fill=-1, tol=.2):
    g = xr.DataArray(round_discard(ds.mz.values, fill=fill, tol=tol), dims=['mz'], coords=[ds.mz], name='g')
    ds = ds.groupby(g).sum(dim='mz').rename({'g':'mz'})
    ds['temp'] = ds.temp.isel(mz=0).drop('mz')
    ds['time'] = ds.time.isel(mz=0).drop('mz')
    return ds

def drop_helium(ds):
    return ds.drop_sel(mz=4)

def drop_heavy(ds, mz_thresh=80):
    return ds.sel(mz=slice(None, mz_thresh))

def append_labels_and_meta(ds):
    df = pd.concat([
        pd.read_csv("../data/val_labels.csv")
            .set_index('sample_id', drop=True),
        pd.read_csv("../data/train_labels.csv")
            .set_index('sample_id', drop=True)
    ])
    
    labels = (
        df.to_xarray()
        .to_array(dim='species', name='labels')
    ).transpose()
    
    metadata = (
        pd.read_csv("../data/metadata.csv")
        .set_index('sample_id', drop=True)
        .to_xarray()
        .drop(['features_path', 'features_md5_hash'])
    )

    ds = ds.merge(labels).merge(metadata)
    return ds

def bin_by_temp(ds, bins):
    ds_new = ds.groupby_bins('temp', bins).mean(dim='index')
    ds_obs = (
        ds.groupby_bins('temp', bins)
        .count(dim='index')
        #.reduce(np.nansum, dim='index')
        .abundance.fillna(0)
        .rename('number_of_obvs')
        .astype(int)
    )
    return ds_new.merge(ds_obs)

def _calculate_binned_data(bins, mz_thresh):

    all_ds = []

    files = sorted(
        glob("../data/train_features/*.csv") +
        glob("../data/val_features/*.csv") +
        glob("../data/test_features/*.csv"),
        key = lambda x: x.split('/')[-1]
    )

    for f in tqdm(files):
        # topu
        df = pd.read_csv(f)
        sample_id = f.split('/')[-1].replace('.csv', '')

        ds_new = reshape_df_to_ds(df)
        ds_new = ds_new.assign_coords({"sample_id": [sample_id]})
        
        ds_new = drop_frac_mz_with_rounding(ds_new)
        ds_new = drop_heavy(ds_new, mz_thresh=mz_thresh)
        ds_new = bin_by_temp(ds_new, bins)
        all_ds.append(ds_new)
    
    ds_regrid = xr.concat(all_ds, dim='sample_id')
    ds_regrid = append_labels_and_meta(ds_regrid)
    return ds_regrid

def rebin_data(ds, bins, mz_thresh):
    
    threshold_valid = mz_thresh <= ds.mz.max()
    
    ds_bin_rights = np.vectorize(lambda x: x.right)(ds.temp_bins.values)
    ds_bin_limits = np.insert(ds_bin_rights, 0, ds.temp_bins.values[0].left)
    bins_valid = np.all(
        np.isin(
            bins, 
            ds_bin_limits, 
            assume_unique=False
        )
    )
    
    if threshold_valid and bins_valid:
        ds_bins_mid = xr.zeros_like(ds.temp_bins)
        ds_bins_mid.values[:] = np.vectorize(lambda x: x.mid)(ds.temp_bins.values)
        ds_counts = ds.number_of_obvs.groupby_bins(ds_bins_mid, bins).reduce(np.nansum)
        ds_abun = ds.abundance*ds.number_of_obvs
        ds_abun =  ds_abun.groupby_bins(ds_bins_mid, bins).reduce(np.nansum) / ds_counts
        ds_t = ds[['temp', 'time']]*ds.number_of_obvs.mean(dim='mz')
        ds_t =  ds_t.groupby_bins(ds_bins_mid, bins).reduce(np.nansum) / ds_counts.mean(dim='mz')
        
        ds_new = (
            xr.merge([
                
                ds[[k for k in ds.keys() if k not in ['abundance', 'number_of_obvs', 'temp', 'time']]], 
                ds_counts.rename('number_of_obvs'), 
                ds_abun.rename('abundance'), 
                ds_t
            ])
            .rename({'temp_bins_bins':'temp_bins'})  
        )
        ds_new = drop_heavy(ds_new, mz_thresh=mz_thresh)
        ds_new = ds_new.transpose('sample_id', 'species', 'temp_bins', 'mz')
        return ds_new
    else:
        warnings.warn(f"Could not load selected bins {bins} and threshold {mz_thresh}.")
        return None    

def _load_binned_data(bins, mz_thresh):
    ds = decode_bins(
        dask.compute(
            xr.open_zarr('../data/temp_binned_1.zarr'), 
            scheduler='threads'
        )[0]
    )
    return rebin_data(ds, bins, mz_thresh)
    

def get_binned_data(bins=BINS, mz_thresh=80, recalculate=False):
    
    if recalculate:
        return _calculate_binned_data(bins, mz_thresh)
    else:
        ds = _load_binned_data(bins, mz_thresh)
        if ds is not None:
            return ds
        else:
            return _calculate_binned_data(bins, mz_thresh)

def encode_bins(ds):
    right = xr.zeros_like(ds.temp_bins)
    right.values[:] = np.vectorize(lambda x: x.right)(ds.temp_bins.values)
    ds['right'] = right
    
    ds = ds.swap_dims({'temp_bins': 'right'})
    
    left = xr.zeros_like(ds.right)
    
    left.values[:] = np.vectorize(lambda x: x.left)(ds.temp_bins.values)
    ds['left'] = left
    
    ds = ds.drop('temp_bins')
    return ds

def decode_bins(ds):
    bins = xr.zeros_like(ds.right, dtype=object)
    bins.values[:] = np.vectorize(lambda left, right: pd.Interval(left, right, closed='right'))(ds.left.values, ds.right.values)
    ds['temp_bins'] = bins
    
    ds = ds.swap_dims({'right':'temp_bins'}).drop(('right', 'left'))
    return ds

def supplementary_append_labels_and_meta(ds):
    metadata = (
        pd.read_csv("../data/supplemental_metadata.csv")
        .set_index('sample_id', drop=True)
        .to_xarray()
        .drop(['features_path', 'features_md5_hash'])
    )

    ds = ds.merge(metadata)
    return ds

def supplementary_get_binned_data(bins=BINS, mz_thresh=80, rough_filter=True):

    all_ds = []

    files = sorted(
        glob("../data/supplemental_features/*.csv"),
        key = lambda x: x.split('/')[-1]
    )

    for f in tqdm(files):
        # topu
        df = pd.read_csv(f)
        sample_id = f.split('/')[-1].replace('.csv', '')

        ds_new = reshape_df_to_ds(df)
        ds_new = ds_new.assign_coords({"sample_id": [sample_id]})
        
        ds_new = drop_frac_mz_with_rounding(ds_new)
        ds_new = drop_heavy(ds_new, mz_thresh=mz_thresh)
        try:
            ds_new = bin_by_temp(ds_new, bins)
            all_ds.append(ds_new)
        except:
            warnings.warn(f"Problem with file: {f}")
    
    ds_regrid = xr.concat(all_ds, dim='sample_id')
    ds_regrid = supplementary_append_labels_and_meta(ds_regrid)
    
    if rough_filter:
        ds_regrid = ds_regrid.where(ds_regrid.carrier_gas=='he', drop=True)
        ds_regrid = ds_regrid.where(ds_regrid.different_pressure==0, drop=True)
        ds_regrid = ds_regrid.drop(('carrier_gas', 'different_pressure'))
    
    return ds_regrid

class SpectroDataset(Dataset):
    
    def __init__(self, ds, get_y=True, augmentation=None):
        self.ds = ds
        self.augmentation = augmentation
        self.get_y = get_y
    
    def __len__(self):
        return self.ds.sample_id.shape[0]

    def __getitem__(self, idx):
        # need to set single-threaded scheduler so no clash with DataLoader scheduler
        da = self.ds.isel(sample_id=idx)
        image = da.features.values[None]
        image = torch.from_numpy(image).float()
        
        if self.augmentation:
            image = self.augmentation(image)
        
        if self.get_y:
            label = da.labels.values
            label = torch.from_numpy(label).float()
            return image, label
        
        else:
            return image


class SpectroDataModule(pl.LightningDataModule):
    def __init__(self, ds, ds_val=None, **kwargs):
        super().__init__()
        
        self.num_workers = kwargs.get("num_workers", 2)
        self.batch_size = kwargs.get("batch_size", 16)
        self.augmentation = kwargs.get("augmentation", None)
    
        if ds_val is None:
            self.val_frac = kwargs.get("val_frac", 0.2)
            self.data_split_seed = kwargs.get("data_split_seed", None)
            train_samples, test_samples = train_test_split(self.ds.sample_id.values, random_state=self.data_split_seed)
            self.ds_train = ds.sel(sample_id=train_samples)
            self.ds_val = ds.sel(sample_id=test_samples)
        else:
            self.ds_train = ds
            self.ds_val = ds_val
            self.val_frac = ds_val.sample_id.shape[0]/(ds.sample_id.shape[0]+ds_val.sample_id.shape[0])
            self.data_split_seed = None


    def setup(self, stage: Optional[str] = None):
        # split dataset
        if stage not in (None, "fit"):
            raise NotImplementedError
        
        self.train_dataset = SpectroDataset(
            self.ds_train,
            augmentation=self.augmentation,
        )
        self.val_dataset = SpectroDataset(
            self.ds_val,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )
    
        