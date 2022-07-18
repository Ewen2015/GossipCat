#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import pandas as pd
import matplotlib.pyplot as plt

from neuralprophet import NeuralProphet, set_log_level
from neuralprophet.utils import set_random_seed
set_log_level("ERROR")
set_random_seed(seed=0)


class NPGlobal(object):
    """docstring for Stratify"""
    def __init__(self, df_dict, stratify=True, df_info=None, stratify_on=None, ts_col=None):
        super(NPGlobal, self).__init__()
        self.df_dict = df_dict

        self.params = {
            'growth': 'discontinuous',
            
            'changepoints': None,
            'n_changepoints': 4,
            'changepoints_range': 0.90,
            'trend_reg': 1,
            'trend_reg_threshold':False,

            'n_lags': 2*3,
            'n_forecasts': 2*3,
            'valid_n': 12,
            'measure': 'RMSE_val',
            
            'loss_func': 'MSE',
            'num_hidden_layers': 2,
            'd_hidden': 4,
            'ar_reg': 0,
            'learning_rate': 0.01,
            'batch_size': None,
            'epochs': None,

            'yearly_seasonality': True,
            'weekly_seasonality': False,
            'daily_seasonality': False,

            'global_normalization': True
        }

        if stratify:
            self.df_info = df_info
            self.stratify_on = stratify_on
            self.ts_col = ts_col
            self.dict_stratify = self.get_stratify_dict(self.df_info, self.stratify_on, self.ts_col)


    def train_test_split(self, df_dict_sub, valid_n=12):
        df_train_dict = dict()
        df_val_dict = dict()

        for k, v in df_dict_sub.items():
            df_train_dict[k] = df_dict_sub[k].iloc[:-valid_n, :].reset_index(drop=True)
            df_val_dict[k] = df_dict_sub[k].tail(valid_n).reset_index(drop=True)

        return df_train_dict, df_val_dict

    def model_and_measure(self, df_dict_sub, progress="plot-all"):
        set_random_seed(seed=0)
        
        m = NeuralProphet(
            growth = self.params['growth'],
            
            changepoints = self.params['changepoints'],
            n_changepoints = self.params['n_changepoints'],
            changepoints_range = self.params['changepoints_range'],
            trend_reg = self.params['trend_reg'],
            trend_reg_threshold = self.params['trend_reg_threshold'],

            n_lags = self.params['n_lags'],

            num_hidden_layers = self.params['num_hidden_layers'],
            d_hidden = self.params['d_hidden'],
            ar_reg = self.params['ar_reg'],
            learning_rate = self.params['learning_rate'],
            loss_func=self.params['loss_func'],

            n_forecasts = self.params['n_forecasts'],

            yearly_seasonality = self.params['yearly_seasonality'],
            weekly_seasonality = self.params['weekly_seasonality'],
            daily_seasonality = self.params['daily_seasonality'],

            global_normalization = self.params['global_normalization']
            )

        df_train_dict, df_val_dict = self.train_test_split(df_dict_sub, valid_n=self.params['valid_n'])
        
        metrics = m.fit(df_train_dict, validation_df=df_val_dict, progress=progress)
        
        measure_best = metrics.sort_values(self.params['measure']).head(1)[self.params['measure']].values[0]
        measure_best = round(measure_best, 2)
        return m, measure_best


    def get_stratify_dict(self, df_info, stratify_on, ts_col):
        dict_stratify = df_info[[stratify_on, ts_col]].drop_duplicates().groupby(stratify_on)[ts_col].apply(list).to_dict()
        return dict_stratify

    def build_with_stratify(self, progress="plot-all"):
        self.dict_count = dict()
        self.dict_measures = dict()
        self.dict_model = dict()

        for f, v in self.dict_stratify.items():  
            
            print('{}: {}'.format(self.stratify_on, f))
            
            df_dict_tmp = dict((k, self.df_dict[k]) for k in v if k in self.df_dict.keys())
            
            cnt = len(df_dict_tmp.keys())
            if cnt == 0:
                print('WARNING: No DataFrames in {} meet the filter rules.'.format(f))
                continue
            else:
                self.dict_count[f] = cnt
            
            self.dict_model[f], self.dict_measures[f] = self.model_and_measure(df_dict_tmp, progress=progress)

        self.df_summary = pd.DataFrame([self.dict_count, self.dict_measures]).transpose().reset_index()
        self.df_summary.columns = [self.stratify_on, 'count', self.params['measure']]

        return self.df_summary

    def search_linear(self, param=None, param_range=None, progress="plot-all"):

        self.dict_exp = dict()

        for i in param_range:
            self.params[param] = i

            _, self.dict_exp[i] = self.model_and_measure(self.df_dict)

        self.exp_summary = pd.DataFrame(self.dict_exp.items(), columns=[param, self.params['measure']])
        return self.exp_summary

    def search_linear_stratify(self, param=None, param_range=None, progress="plot-all"):

        self.exp_summary = pd.DataFrame()

        for i, v in enumerate(param_range):
            self.params[param] = v

            tmp = self.build_with_stratify(progress=progress)

            if i == 0:
                self.exp_summary = tmp 
            else:
                self.exp_summary = pd.concat([self.exp_summary, tmp[[self.params['measure']]]], axis=1)

            self.exp_summary.columns = [*self.exp_summary.columns[:-1], '_'.join([param, str(v)])]

        return self.exp_summary