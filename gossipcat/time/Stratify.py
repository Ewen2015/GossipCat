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


class Stratify(object):
    """docstring for Stratify"""
    def __init__(self, df_dict, df_info, feature, ts):
        super(Stratify, self).__init__()
        self.df_dict = df_dict
        self.df_info = df_info
        self.feature = feature
        self.ts = ts
        self.dict_feature = self.get_feature_dict(self.df_info, self.feature, self.ts)

        self.params = {
            'growth': 'discontinuous',

            'n_lags': 2*3,

            'num_hidden_layers': 2,
            'd_hidden': 2,
            'learning_rate': 0.01,

            'n_forecasts': 2*3,

            'yearly_seasonality': True,
            'weekly_seasonality': False,
            'daily_seasonality': False,

            'n_changepoints': 4,
            'changepoints_range': 0.90,
            'trend_reg': 1,

            'global_normalization': True
        }

    def train_test_split(self, valid_n=12):
        df_train_dict = dict()
        df_val_dict = dict()

        for k, v in df_dict.items():
            df_train_dict[k] = df_dict[k].iloc[:-valid_n, :].reset_index(drop=True)
            df_val_dict[k] = df_dict[k].tail(valid_n).reset_index(drop=True)

        return df_train_dict, df_val_dict

    def model_and_measure(self, df_dict, valid_n=12, progress="plot-all", measure='RMSE_val'):
        set_random_seed(seed=0)
        
        m = NeuralProphet(self.params)

        df_train_dict, df_val_dict = self.train_test_split(df_dict, valid_n=valid_n)
        
        metrics = m.fit(df_train_dict, validation_df=df_val_dict, progress=progress)
        
        measure_best = metrics.sort_values(measure).head(1)[measure].values[0]
        measure_best = round(measure_best, 2)
        return m, measure_best


    def get_feature_dict(self, df_info, feature, ts):
        self.dict_feature = self.df_info[[self.feature, self.ts]].drop_duplicates().groupby(self.feature)[self.ts].apply(list).to_dict()
        return self.dict_feature

    def build_all(self, valid_n=12, progress="plot-all", measure='RMSE_val'):

        self.dict_count = dict()
        self.dict_measures = dict()
        self.dict_model = dict()

        for f, v in self.dict_feature.items():  
            
            print('{}: {}'.format(feature, f))
            
            self.df_dict = dict((k, self.df_dict[k]) for k in v)
            
            cnt = len(self.df_dict.keys())
            if cnt == 0:
                print('WARNING: No DataFrames in {} meet the filter rules.'.format(f))
                continue
            else:
                self.dict_count[f] = cnt
            
            self.dict_model[f], self.dict_measures[f] = self.model_and_measure(self.df_dict, valid_n=valid_n, progress=progress, measure=progress)

        self.df_summary = pd.DataFrame([self.dict_measures, self.dict_count]).transpose().reset_index()
        self.df_summary.columns = [self.feature, measure, 'count']

        return self.df_summary





