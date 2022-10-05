import numpy as np
import pandas as pd
import xarray as xr
import numba as nb
import dask
from dask.diagnostics import ProgressBar


import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, log_loss
from sklearn.model_selection import ParameterSampler
from sklearn.utils import shuffle
import sklearn

from xgboost import XGBClassifier
import pytorch_tabnet
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

from glob import glob
from tqdm import tqdm
import itertools
import time
import os
import copy

import optuna

from typing import Optional, List, Union
import logging
from IPython.utils import io

import spect

scorer = make_scorer(log_loss, needs_proba=True)

missing_fill = np.nan


def model_save_callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_model", value=trial.user_attrs["model"])
        
def default_parameter_sampler(trial):
    params = {
        "n_estimators": trial.suggest_int('n_estimators', 300, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate',0.005,0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        
        'subsample': trial.suggest_float('subsample', 0.5, 1.),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.),
        
        'gamma': trial.suggest_loguniform('gamma', 0.001, .5),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 5.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 5.0),
        
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 0.005, 1),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
        'max_delta_step': trial.suggest_float('max_delta_step', 0, 10),
   }
    return params


class XG:
    def __init__(self, patience=10, **kwargs):
        self.patience = patience
        self._single_xgboost = XGBClassifier(
            n_jobs=20,
            eval_metric='logloss',
            gpu_id=0,
            use_label_encoder=False,
            missing=missing_fill,
            tree_method='exact',
            num_parallel_tree=5,
            **kwargs,
        )
                
        self.xgboost = []
        self.optuna_studies = []
        self.logloss_report = None
        
    def _fit_and_score(self, clf, X_train, y_train, X_val, y_val, train_params=None):
        clf.fit(
            X_train, y_train,
            eval_set=((X_train, y_train), (X_val, y_val),),
            early_stopping_rounds=self.patience,
            eval_metric="logloss",
            verbose=0
        )
        s = scorer(clf, X_val, y_val)
        return clf, s
    
    def _prepare_data_xgboost(self, ds, get_y=True):
        X = [
            ds.features.stack(dict(z=("temp_bins","mz"))).values, 
            (ds.instrument_type.values=='commercial').astype(float)[:, None],
            ds.integrated_abundance.values,
            ds.time.values,
        ]
        X = np.concatenate(X, axis=1)
        
        if get_y:
            y = ds.labels.values
            return X, y
        else:
            return X
        
    def _fit_individual_columns(self, ds_train, ds_val, parameter_sampler, n_trials):
                
        X_train, y_train_all = self._prepare_data_xgboost(ds_train)
        X_val, y_val_all = self._prepare_data_xgboost(ds_val)
        
        def objective(trial):
            params = parameter_sampler(trial)
            model = sklearn.base.clone(self._single_xgboost).set_params(**params)
            model, s = self._fit_and_score(model, X_train, y_train, X_val, y_val)            
            trial.set_user_attr(key="model", value=model)
            
            return s
        
        for col in range(y_train_all.shape[1]):
            y_train = y_train_all[:, col]
            y_val = y_val_all[:, col]

            study = optuna.create_study(
                #pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), 
                direction="minimize",
            )
            study.optimize(objective, n_trials=n_trials, callbacks=[model_save_callback], show_progress_bar=True)
            self.optuna_studies += [study]
            self.xgboost.append(study.user_attrs["best_model"])
        
        self.logloss_report = self.estimate_logloss(ds_train, ds_val)
        
        return
    
    def _fit_same_params_all_columns(self, ds_train, ds_val, parameter_sampler, n_trials):
                
        X_train, y_train_all = self._prepare_data_xgboost(ds_train)
        X_val, y_val_all = self._prepare_data_xgboost(ds_val)
        
        def objective(trial):
            params = parameter_sampler(trial)
            r = []
            for col in range(y_train_all.shape[1]):
                y_train = y_train_all[:, col]
                y_val = y_val_all[:, col]
                
                model = sklearn.base.clone(self._single_xgboost).set_params(**params)
                r += [dask.delayed(self._fit_and_score)(model, X_train, y_train, X_val, y_val)]
                
            models = []
            s = 0
            for model, s_ in dask.compute(*r, scheduler='threads'):
                models += [model]
                s = s + s_
                
            s = s/y_train_all.shape[1]
            trial.set_user_attr(key="models", value=models)
            
            return s

        study = optuna.create_study(
            #pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), 
            direction="minimize",
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        self.optuna_study = study
        self.xgboost = study.best_trial.user_attrs["models"]
        
        self.logloss_report = self.estimate_logloss(ds_train, ds_val)
        
        return
    
    def fit(self, ds_train, ds_val, parameter_sampler=default_parameter_sampler, same_params_all_columns=False, n_trials=10):
        self.species = ds_train.species.values

        if same_params_all_columns:
            return self._fit_same_params_all_columns(ds_train, ds_val, parameter_sampler, n_trials)
        else:
            return self._fit_individual_columns(ds_train, ds_val, parameter_sampler, n_trials)
    
    def _predict_proba(self, ds):
        X = self._prepare_data_xgboost(ds, get_y=False)
        y_preds = []
        for col in range(self.species.shape[0]):
            y_preds += [self.xgboost[col].predict_proba(X)[:, 1:]]
        return np.concatenate(y_preds, axis=1)
    
    def predict_proba(self, ds):
        y_preds = self._predict_proba(ds)
        ds_pred = xr.DataArray(
            y_preds, 
            dims=['sample_id', 'species'], 
            coords=[ds.sample_id.values, self.species], 
            name='preds'
        )
        return ds_pred
    
    def estimate_logloss(self, ds_train, ds_val):
        boosted_train = xr_loss(self.predict_proba(ds_train), ds_train.labels)
        boosted_val = xr_loss(self.predict_proba(ds_val), ds_val.labels)
        
        losses = xr.merge([
            boosted_train.to_dataset(name='boosted_train'),
            boosted_val.to_dataset(name='boosted_val'),
        ])
        return losses
    
def xr_loss(ds_pred, ds_target):
    return -(ds_target*np.log(ds_pred)+(1-ds_target)*np.log(1-ds_pred)).mean(dim=('sample_id'))



def default_parameter_sampler2(trial):
    params = {
        "n_estimators": trial.suggest_int('n_estimators', 300, 1000, step=10), ##test
        'learning_rate': trial.suggest_loguniform('learning_rate',0.005,0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        
        'subsample': trial.suggest_float('subsample', 0.5, 1.),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.),
        
        'gamma': trial.suggest_loguniform('gamma', 0.001, .5),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 5.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 5.0),
        
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 0.005, 1),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 20),
        'max_delta_step': trial.suggest_float('max_delta_step', 0, 10),
        
        # to load the data
        'data_mz_threshold': trial.suggest_int('mz_threshold', 50, 200),
        'data_bin_width': trial.suggest_int('bin_width', 30, 500, step=1, log=True),
        'data_bin_max': trial.suggest_int('bin_max', 400, 1490, step=10),
        
        # filter mz values
        'data_abundance_thresh': trial.suggest_loguniform('abundance_thresh', 1e-9, 1.),
        
        # preprocess
        'data_drop_he': trial.suggest_categorical("drop_he", [True, False]),
        'data_rebase': trial.suggest_categorical("rebase", [True, False]),
        'data_maxscale': trial.suggest_categorical("maxscale", [True, False]),
        'data_intmaxscale': trial.suggest_categorical("intmaxscale", [True, False]),
        'data_logscale': trial.suggest_categorical("logscale", [True, False]),
   }
    return params

def print_ds(ds):
    print(ds.isel(sample_id=slice(0,10)))
    
class XG2(XG):

    def filter_mz(self, ds, max_thresh=0., **kwargs):
        da = ds.abundance.where(ds.instrument_type=='commercial', drop=True)
        da = da.max(dim=('sample_id', 'temp_bins'))
        thresh = max_thresh*da.max()
        thresh = max_thresh*da.max().item()
        return ds.sel(mz=(da>=thresh))

    def preprocess(self, ds, data_drop_he=False, data_rebase=False, data_maxscale=True, data_intmaxscale=True, nan_fill=missing_fill, **kwargs):
        ds_pro = ds
        ds_pro['features'] = ds_pro['abundance'].clip(0, None)
        ds_pro['integrated_abundance'] = ds_pro['features'].mean(dim='temp_bins')

        if data_rebase:
            ds_pro['features'] = ds_pro.features - ds_pro.features.min(dim=('temp_bins'))
        if data_drop_he:
            ds_pro = ds_pro.drop_sel(mz=4)
        if data_maxscale:
            ds_pro['features'] = ds_pro.features/ds_pro.features.max(dim=('temp_bins', 'mz'))
        if data_intmaxscale:
            ds_pro['integrated_abundance'] = ds_pro.integrated_abundance/ds_pro.integrated_abundance.max(dim='mz')
        ds_pro = ds_pro.fillna(nan_fill)
        return ds_pro
        
    def prepare_datasets(self, ds_train=None, ds_val=None, params=None):
        assert (ds_train is not None) or self.is_fit
        
        def data_pipe(ds):
            ds = spect.rebin_data(
                ds,
                bins=np.arange(0, params['data_bin_max']+1, params['data_bin_width']), 
                mz_thresh=params['data_mz_threshold']
            )
            ds = self.preprocess(ds, **params)
            return ds
                
        r = []
        if ds_train is not None:
            ds_train = data_pipe(ds_train)
            if not self.is_fit:
                ds_train = self.filter_mz(ds_train, max_thresh=params['data_abundance_thresh'])
            else:
                ds_train = ds_train.sel(mz=params['data_mz_values'])
            r += [ds_train]

        
        if ds_val is not None:
            ds_val = data_pipe(ds_val)
            if not self.is_fit:
                ds_val = ds_val.sel(mz=ds_train.mz.values)
            else:
                ds_val = ds_val.sel(mz=params['data_mz_values'])
            r += [ds_val]
            
        return r
        
    def _fit_individual_columns(self, ds_train, ds_val, parameter_sampler, n_trials):
        
        def objective(trial):
            params = parameter_sampler(trial)
            
            ds_train, ds_val = self.prepare_datasets(ds_train=ds_train, ds_val=ds_val, params=params)
            
            X_train, y_train_all = self._prepare_data_xgboost(ds_train)
            X_val, y_val_all = self._prepare_data_xgboost(ds_val)
            y_train = y_train_all[:, col]
            y_val = y_val_all[:, col]
            
            model_params = {k:v for k,v in params.items() if 'data_' not in k}
            
            model = sklearn.base.clone(self._single_xgboost).set_params(**model_params)
            model, s = self._fit_and_score(model, X_train, y_train, X_val, y_val)
            
            trial.set_user_attr(key="mz_values", value=ds_train.mz.values)
            trial.set_user_attr(key="model", value=model)
            trial.set_user_attr(key="params", value=params)
            
            return s
        
        params = []
        
        for col in range(y_train_all.shape[1]):

            study = optuna.create_study(
                #pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), 
                direction="minimize",
            )
            study.optimize(objective, n_trials=n_trials, callbacks=[model_save_callback], show_progress_bar=True)
            self.optuna_studies += [study]
            self.xgboost += [study.best_trial.user_attrs["model"]]
            p = study.best_trial.user_attrs["params"]
            p['data_mz_values'] = study.best_trial.user_attrs["mz_values"]
            params += [p]
            
        self.params = params
        
        return
    
    def _fit_same_params_all_columns(self, ds_train, ds_val, parameter_sampler, n_trials):

        def objective(trial):
            params = parameter_sampler(trial)
            
            this_ds_train, this_ds_val = self.prepare_datasets(ds_train=ds_train, ds_val=ds_val, params=params)
            X_train, y_train_all = self._prepare_data_xgboost(this_ds_train)
            X_val, y_val_all = self._prepare_data_xgboost(this_ds_val)
            
            r = []
            for col in range(y_train_all.shape[1]):
                y_train = y_train_all[:, col]
                y_val = y_val_all[:, col]
                
                model_params = {k:v for k,v in params.items() if 'data_' not in k}
                model = sklearn.base.clone(self._single_xgboost).set_params(**model_params)
                r += [dask.delayed(self._fit_and_score)(model, X_train, y_train, X_val, y_val)]
                
            models = []
            s = 0
            for model, s_ in dask.compute(*r, scheduler='threads'):
                models += [model]
                s = s + s_
                
            s = s/y_train_all.shape[1]
            trial.set_user_attr(key="mz_values", value=this_ds_train.mz.values)
            trial.set_user_attr(key="models", value=models)
            trial.set_user_attr(key="params", value=params)
            
            return s

        study = optuna.create_study(
            #pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), 
            direction="minimize",
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        self.optuna_study = study
        
        self.xgboost = study.best_trial.user_attrs["models"]
        self.params = study.best_trial.user_attrs["params"]
        self.params['data_mz_values'] = study.best_trial.user_attrs["mz_values"]
        
        return
    
    def drop_studies(self):
        self.optuna_studies = None
        self.optuna_study = None
    
    def fit(self, ds_train, ds_val, parameter_sampler=default_parameter_sampler2, same_params_all_columns=False, n_trials=10):
        self.is_fit = False
        
        self.species = ds_train.species.values

        if same_params_all_columns:
            self.share_params = True
            r =  self._fit_same_params_all_columns(ds_train, ds_val, parameter_sampler, n_trials)
        else:
            self.share_params = False
            r = self._fit_individual_columns(ds_train, ds_val, parameter_sampler, n_trials)

        self.is_fit = True
        self.logloss_report = self.estimate_logloss(ds_train, ds_val)
        
        return r
    
    def _predict_proba(self, ds):
        if self.share_params:
            ds, = self.prepare_datasets(ds_train=None, ds_val=ds, params=self.params)
            X = self._prepare_data_xgboost(ds, get_y=False)
        y_preds = []
        for col in range(self.species.shape[0]):
            if not self.share_params:
                ds, = self.prepare_datasets(ds_train=None, ds_val=ds, params=self.params[col])
                X = self._prepare_data_xgboost(ds, get_y=False)
            y_preds += [self.xgboost[col].predict_proba(X)[:, 1:]]
        return np.concatenate(y_preds, axis=1)


class cv_ensemble:
    def __init__(self, models, preprocessors):
        self.models = models
        self.preprocessors = preprocessors
    
    def predict_proba(self, ds, reduce=np.mean):
        @dask.delayed
        def f(ds, model, preproc):
            X = preproc.transform(ds, get_y=False)
            return model.predict_proba(X)[:, 1:]
        
        y_preds = []
        for model, preproc in zip(self.models, self.preprocessors):
            y_preds += [f(ds, model, preproc)]
        with ProgressBar(dt=1):
            y_preds = dask.compute(y_preds, scheduler='threads')[0]
        y_preds = np.stack(y_preds, axis=2)
        if reduce is not None:
            return np.mean(y_preds, axis=2)
        else:
            return y_preds
    
class Preprocessor:
    def __init__(self, params, use_preds=False):
        self.is_fit = False
        self.params=params
        self.use_preds=use_preds
        
    def filter_mz(self, ds, max_thresh=0., **kwargs):
        da = ds.abundance.where(ds.instrument_type=='commercial', drop=True)
        da = da.max(dim=('sample_id', 'temp_bins'))
        thresh = max_thresh*da.max()
        thresh = max_thresh*da.max().item()
        return ds.sel(mz=(da>=thresh))

    def preprocess(self, ds, data_drop_he=False, data_rebase=False, data_maxscale=True, data_logscale=False, data_nan_fill=missing_fill, **kwargs):
        ds_pro = ds
        ds_pro['features'] = ds_pro['abundance'].clip(0, None)
        ds_pro['integrated_abundance'] = ds_pro['features'].mean(dim='temp_bins')

        if data_rebase:
            ds_pro['features'] = ds_pro.features - ds_pro.features.min(dim=('temp_bins'))
        if data_drop_he:
            ds_pro = ds_pro.drop_sel(mz=4)
        if data_maxscale:
            ds_pro['features'] = ds_pro.features/ds_pro.features.max(dim=('temp_bins', 'mz'))
        if data_logscale:
            ds_pro['features'] = np.log(ds_pro.features.clip(1e-16, None))
            ds_pro['integrated_abundance'] = np.log(ds_pro.integrated_abundance.clip(1e-16, None))
        ds_pro = ds_pro.fillna(data_nan_fill)
        return ds_pro
    
    def transform(self, ds, get_y=True):
        assert self.is_fit
        
        ds = spect.rebin_data(
            ds,
            bins=np.arange(0, self.params['data_bin_max']+1, self.params['data_bin_width']), 
            mz_thresh=self.params['data_mz_threshold']
        )
        ds = self.preprocess(ds, **self.params)
        ds = ds.sel(mz=self.mz)
            
        return self._prepare_data_xgboost(ds, get_y)
    
    def _prepare_data_xgboost(self, ds, get_y=True):
        X = [
            ds.features.stack(dict(z=("temp_bins","mz"))).values, 
            (ds.instrument_type.values=='commercial').astype(float)[:, None],
            ds.integrated_abundance.values,
            ds.time.values,
        ]
        if self.use_preds:
            X+=[ds.preds.values]
        X = np.concatenate(X, axis=1)
        
        if get_y:
            y = ds.labels.values
            return X, y
        else:
            return X
    
    def fit_transform(self, ds, get_y=True):
        ds = spect.rebin_data(
            ds,
            bins=np.arange(0, self.params['data_bin_max']+1, self.params['data_bin_width']), 
            mz_thresh=self.params['data_mz_threshold']
        )
        ds = self.preprocess(ds, **self.params)
        ds = self.filter_mz(ds, max_thresh=self.params['data_abundance_thresh'])
        self.mz = ds.mz.values
        self.is_fit = True
        return self._prepare_data_xgboost(ds, get_y)
        
class XG2CV(XG2):
    
    def __init__(self, patience=10, study_name=None, use_preds=False, **kwargs):
        
        self.patience = patience
        self._single_xgboost = XGBClassifier(
            n_jobs=20,
            eval_metric='logloss',
            gpu_id=0,
            use_label_encoder=False,
            missing=missing_fill,
            tree_method='exact',
            num_parallel_tree=5,
            **kwargs,
        )
        
        self.study_name = study_name
        self.xgboost = []
        self.optuna_studies = []
        self.logloss_report = None
        self.use_preds = use_preds

    def _fit_individual_columns(self, ds_splits, ds_sup, parameter_sampler, n_trials):
        
        storage_root = "/home/s1205782/Datastore/Projects/spect/notebooks/nb11_results"
        
        def objective_col(col):
            def objective(trial):
                params = parameter_sampler(trial)

                preprocs = []
                models = []
                s = 0

                for i in range(len(ds_splits)):

                    ds_val = ds_splits[i]
                    ds_train = [ds for j, ds in enumerate(ds_splits) if j!=i]
                    if ds_sup is not None:
                        ds_train += [ds_sup]
                    ds_train = xr.concat(ds_train, dim='sample_id', data_vars="all")

                    preproc = Preprocessor(params=params, use_preds=self.use_preds)
                    X_train, y_train_all = preproc.fit_transform(ds_train)
                    X_val, y_val_all = preproc.transform(ds_val)

                    y_train = y_train_all[:, col]
                    y_val = y_val_all[:, col]

                    model_params = {k:v for k,v in params.items() if 'data_' not in k}

                    model = sklearn.base.clone(self._single_xgboost).set_params(**model_params)

                    model, s_ = self._fit_and_score(model, X_train, y_train, X_val, y_val)
                    preprocs += [preproc]
                    models += [model]
                    s += s_

                    step = i+1
                    intermediate_value = s_ # s/step
                    trial.report(intermediate_value, step)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

                s = s/len(ds_splits)

                model = cv_ensemble(
                        models=models,
                        preprocessors=preprocs,
                )

                trial.set_user_attr(key="model", value=model)
                trial.set_user_attr(key="params", value=params)

                return s
            return objective
        
        
        @dask.delayed
        def fit_col(col):
            
            objective = objective_col(col)
            
            study = optuna.create_study(
                study_name=self.study_name,
                storage=None if self.study_name is None else f"{storage_root}/{self.study_name}-col-{col}",
                pruner=optuna.pruners.PercentilePruner(
                    50.0, n_startup_trials=10, n_warmup_steps=2, interval_steps=1, n_min_trials=10, ##test
                ),
                direction="minimize",
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=(col==0))
            
            model = study.best_trial.user_attrs["model"]
            params = study.best_trial.user_attrs["params"]
            
            return study, model, params
        
        results = []
        for col in range(len(self.species)):
            results += [fit_col(col)]
            
        results = dask.compute(results, scheduler='threads')[0]
        
        params = []
        for study, model, p in results:
            self.optuna_studies += [study]
            self.xgboost += [model]
            params += [p]
            
        self.params = params
        
        return
    
    def _fit_same_params_all_columns(self, ds_splits, ds_sup, parameter_sampler, n_trials):

        def objective(trial):
            params = parameter_sampler(trial)
            s = 0
            
            cv_models = []
            cv_preprocessors = []
            
            for i in range(len(ds_splits)):
                
                ds_val = ds_splits[i]
                ds_train = [ds for j, ds in enumerate(ds_splits) if j!=i]
                if ds_sup is not None:
                    ds_train += [ds_sup]
                ds_train = xr.concat(ds_train, dim='sample_id', data_vars="all")
                
                preproc = Preprocessor(params=params, use_preds=self.use_preds)
                X_train, y_train_all = preproc.fit_transform(ds_train)
                X_val, y_val_all = preproc.transform(ds_val)

                r = []
                for col in range(y_train_all.shape[1]):
                    y_train = y_train_all[:, col]
                    y_val = y_val_all[:, col]

                    model_params = {k:v for k,v in params.items() if 'data_' not in k}
                    model = sklearn.base.clone(self._single_xgboost).set_params(**model_params)
                    r += [dask.delayed(self._fit_and_score)(model, X_train, y_train, X_val, y_val)]

                models = []
                
                si = 0
                for model, s_ in dask.compute(*r, scheduler='threads'):
                    models += [model]
                    si += s_
                s += si
                step = i+1
                intermediate_value = si/(y_train_all.shape[1])
                trial.report(intermediate_value, step)

                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                cv_models += [models]
                cv_preprocessors += [preproc]
            
            s = s/(y_train_all.shape[1]*len(ds_splits))
                
            cv_models = [
                cv_ensemble(
                    models=[cv[k] for cv in cv_models],
                    preprocessors=[p for p in cv_preprocessors],
                ) 
                for k in range(y_train_all.shape[1])
            ]
            
            trial.set_user_attr(key="models", value=cv_models)
            trial.set_user_attr(key="params", value=params)
            
            return s
        
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.study_name,
            pruner=optuna.pruners.PercentilePruner(
                50.0, n_startup_trials=10, n_warmup_steps=2, interval_steps=1, n_min_trials=10,
            ),
            direction="minimize",
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        self.optuna_study = study
        
        self.xgboost = study.best_trial.user_attrs["models"]
        self.params = study.best_trial.user_attrs["params"]
        
        return
    
    def drop_studies(self):
        self.optuna_studies = None
        self.optuna_study = None
    
    def fit(self, ds_splits, ds_sup=None, parameter_sampler=default_parameter_sampler2, same_params_all_columns=False, n_trials=10):
        self.is_fit = False
        
        self.species = ds_splits[0].species.values

        if same_params_all_columns:
            self.share_params = True
            r =  self._fit_same_params_all_columns(ds_splits, ds_sup, parameter_sampler, n_trials)
        else:
            self.share_params = False
            r = self._fit_individual_columns(ds_splits, ds_sup, parameter_sampler, n_trials)

        self.is_fit = True
        self.logloss_report = 0
        
        return r
    
    def _predict_proba(self, ds, reduce=np.mean):
        y_preds = []
        for cv_pipeline in self.xgboost:
            y_preds += [cv_pipeline.predict_proba(ds, reduce=reduce)]
        return np.concatenate(y_preds, axis=1)
    
    def predict_proba(self, ds, reduce='mean'):
        y_preds = self._predict_proba(ds, reduce=None)
        ds_pred = xr.DataArray(
            y_preds, 
            dims=['sample_id', 'species', 'split'], 
            coords=[ds.sample_id.values, ds.species, np.arange(y_preds.shape[2])], 
            name='preds'
        )
        if reduce is 'mean':
            ds_pred = ds_pred.mean(dim='split')
        return ds_pred
    
    
class XG2CVsplit(XG2CV):

    def _fit_individual_columns(self, ds, folds, kfold_random_seed, ds_sup, parameter_sampler, n_trials):
        
        storage_root = "/home/s1205782/Datastore/Projects/spect/notebooks/nb11_results"
        
        def objective_col(col):
            def objective(trial):
                params = parameter_sampler(trial)

                preprocs = []
                models = []
                s = 0
                
                ds_splits = self.kfold(ds, folds, col, random_state=kfold_random_seed)

                for i in range(folds):
                    
                    ds_val = ds_splits[i]
                    ds_train = [ds for j, ds in enumerate(ds_splits) if j!=i]
                    if ds_sup is not None:
                        ds_train += [ds_sup]
                    ds_train = xr.concat(ds_train, dim='sample_id', data_vars="all")

                    preproc = Preprocessor(params=params, use_preds=self.use_preds)
                    X_train, y_train_all = preproc.fit_transform(ds_train)
                    X_val, y_val_all = preproc.transform(ds_val)                    

                    y_train = y_train_all[:, col]
                    y_val = y_val_all[:, col]

                    train_params = {k:v for k,v in params.items() if 'train_' in k}
                    model_params = {k:v for k,v in params.items() if 'data_' not in k and 'train_' not in k}

                    model = sklearn.base.clone(self._single_xgboost).set_params(**model_params)
                    

                    model, s_ = self._fit_and_score(model, X_train, y_train, X_val, y_val, train_params=train_params)
                    preprocs += [preproc]
                    models += [model]
                    s += s_

                    step = i+1
                    intermediate_value = s_ # s/step
                    trial.report(intermediate_value, step)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

                s = s/len(ds_splits)

                model = cv_ensemble(
                        models=models,
                        preprocessors=preprocs,
                )

                trial.set_user_attr(key="model", value=model)
                trial.set_user_attr(key="params", value=params)

                return s
            return objective
        
        
        @dask.delayed
        def fit_col(col):
            
            objective = objective_col(col)
            
            study = optuna.create_study(
                study_name=self.study_name,
                storage=None if self.study_name is None else f"{storage_root}/{self.study_name}-col-{col}",
                pruner=optuna.pruners.PercentilePruner(
                    50.0, n_startup_trials=10, n_warmup_steps=2, interval_steps=1, n_min_trials=10, ##test
                ),
                direction="minimize",
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=(col==0))
            
            model = study.best_trial.user_attrs["model"]
            params = study.best_trial.user_attrs["params"]
            
            return study, model, params
        
        results = []
        for col in range(len(self.species)):
            results += [fit_col(col)]
            
        results = dask.compute(results, scheduler='threads')[0]
        
        params = []
        for study, model, p in results:
            self.optuna_studies += [study]
            self.xgboost += [model]
            params += [p]
            
        self.params = params
        
        return
    
    def kfold(self, ds, folds, by_species, random_state=None):
        skf = sklearn.model_selection.StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        indices = [v[1] for v in skf.split(ds.sample_id.values, ds.labels.isel(species=by_species).values)]
        ds_splits = [ds.isel(sample_id=inds) for inds in indices]
        return ds_splits
    
    def _fit_same_params_all_columns(self, ds, folds, kfold_random_seed, ds_sup, parameter_sampler, n_trials):

        def objective(trial):
            params = parameter_sampler(trial)
            s = 0
            
            cv_models = []
            cv_preprocessors = []
            
            for i in range(folds):
                
                r = []
                preprocs = []
                
                for col in range(len(self.species)):
                    
                    ds_splits = self.kfold(ds, folds, col, random_state=kfold_random_seed)
                    ds_val = ds_splits[i]
                    ds_train = [ds for j, ds in enumerate(ds_splits) if j!=i]
                    if ds_sup is not None:
                        ds_train += [ds_sup]
                    ds_train = xr.concat(ds_train, dim='sample_id', data_vars="all")
                
                    preproc = Preprocessor(params=params, use_preds=self.use_preds)
                    X_train, y_train_all = preproc.fit_transform(ds_train)
                    X_val, y_val_all = preproc.transform(ds_val)
                    
                    y_train = y_train_all[:, col]
                    y_val = y_val_all[:, col]
                    
                    train_params = {k:v for k,v in params.items() if 'train_' in k}
                    model_params = {k:v for k,v in params.items() if 'data_' not in k and 'train_' not in k}

                    model = sklearn.base.clone(self._single_xgboost).set_params(**model_params)
                    r += [dask.delayed(self._fit_and_score)(model, X_train, y_train, X_val, y_val, train_params=train_params)]
                    preprocs += [preproc]

                models = []
                
                si = 0
                for model, s_ in dask.compute(*r, scheduler='threads'):
                    models += [model]
                    si += s_
                s += si
                step = i+1
                intermediate_value = si/(y_train_all.shape[1])
                trial.report(intermediate_value, step)

                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                cv_models += [models]
                cv_preprocessors += [preprocs]
            
            s = s/(y_train_all.shape[1]*len(ds_splits))
                
            cv_models = [
                cv_ensemble(
                    models=[cv[k] for cv in cv_models],
                    preprocessors=[cv[k] for cv in cv_preprocessors],
                ) 
                for k in range(y_train_all.shape[1])
            ]
            
            trial.set_user_attr(key="models", value=cv_models)
            trial.set_user_attr(key="params", value=params)
            
            return s
        
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.study_name,
            pruner=optuna.pruners.PercentilePruner(
                50.0, n_startup_trials=10, n_warmup_steps=2, interval_steps=1, n_min_trials=10,
            ),
            direction="minimize",
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        self.optuna_study = study
        
        self.xgboost = study.best_trial.user_attrs["models"]
        self.params = study.best_trial.user_attrs["params"]
        
        return
    
    def fit(self, ds, folds, ds_sup=None, parameter_sampler=default_parameter_sampler2, kfold_random_seed=None, same_params_all_columns=False, n_trials=10):
        self.is_fit = False
        
        self.species = ds.species.values

        if same_params_all_columns:
            self.share_params = True
            r =  self._fit_same_params_all_columns(ds, folds, kfold_random_seed, ds_sup, parameter_sampler, n_trials)
        else:
            self.share_params = False
            r = self._fit_individual_columns(ds, folds, kfold_random_seed, ds_sup, parameter_sampler, n_trials)

        self.is_fit = True
        self.logloss_report = 0
        
        return r
    
    
def default_tabnet_parameter_sampler(trial):
    params = {
        "n_a": trial.suggest_int('n_a', 4, 32), ##test
        #"n_d": trial.suggest_int('n_d', 4, 32),
        'gamma': trial.suggest_float('gamma', 1., 2.),
        'n_independent': trial.suggest_int('n_independent', 1, 4),
        'n_shared': trial.suggest_int('n_shared', 1, 4),
        'mask_type': trial.suggest_categorical('mask_type', ['entmax', "sparsemax"]),
        
        # training params
        'train_batch_size': 16, #trial.suggest_int('train_batch_size', 4, 64, log=True),
        
        # to load the data
        'data_mz_threshold': 100, #trial.suggest_int('mz_threshold', 50, 200),
        'data_bin_width': 50, #trial.suggest_int('bin_width', 30, 500, step=1, log=True),
        'data_bin_max': 1000, #trial.suggest_int('bin_max', 400, 1490, step=10),
        
        # filter mz values
        'data_abundance_thresh': 0,# trial.suggest_loguniform('abundance_thresh', 1e-9, 1.),
        
        # preprocess
        'data_drop_he': trial.suggest_categorical("drop_he", [True, False]),
        'data_rebase': False, #trial.suggest_categorical("rebase", [True, False]),
        'data_maxscale': True, #trial.suggest_categorical("maxscale", [True, False]),
        'data_logscale': trial.suggest_categorical("logscale", [True, False]),
        'data_nan_fill': trial.suggest_categorical("data_nan_fill", [-10, -5, -1, 0]),
   }
    params['n_d']=params['n_a']
    return params
    
class TabNetCVsplit(XG2CVsplit):
    
    def __init__(self, patience=10, study_name=None, use_preds=False, **kwargs):

        self.patience = patience
        self._single_xgboost = TabNetClassifier(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-3),
            scheduler_params={
                "step_size":10, # how to use learning rate scheduler
                "gamma":0.9
            },
            verbose=0,
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            seed=546546,
            device_name='cuda',
        )
                
        self.study_name = study_name
        self.xgboost = []
        self.optuna_studies = []
        self.logloss_report = None
        self.use_preds = use_preds
        
    def _fit_and_score(self, clf, X_train, y_train, X_val, y_val, train_params):
        clf.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val),],
            eval_name=['train', 'val'],
            eval_metric=["logloss",],
            patience=self.patience,
            max_epochs=1000, 
            batch_size=train_params['train_batch_size'],
            virtual_batch_size=train_params['train_batch_size'],
            num_workers=0,
            weights=1,
            drop_last=False,
        )
        s = scorer(clf, X_val, y_val)
        return clf, s
    
    def fit(self, ds, folds, ds_sup=None, parameter_sampler=default_tabnet_parameter_sampler, kfold_random_seed=None, same_params_all_columns=False, n_trials=10):
        return super().fit(ds, folds, ds_sup, parameter_sampler, kfold_random_seed, same_params_all_columns, n_trials)
