#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import pandas as pd 

class TimeSeries(object):
    """docstring for TimeSeries"""
    def __init__(self, df, col_date):
        super(TimeSeries, self).__init__()
        self.df = df
        self.col_date = col_date

        # set date column as index
        self.df[self.col_date] = pd.to_datetime(self.df[self.col_date])
        self.df.index = self.df[self.col_date]
        del self.df[self.col_date]
        self.df = self.df.replace(0, None)

        self.columns = self.df.columns

    def to_df_dict(self):
        self.df.insert(loc=0, column='ds', value=self.df.index)
        self.df_list = list()
        self.df_dict = dict()

        for col in self.columns:
            aux = self.df[['ds', col]].copy() 
            aux = aux.rename(columns = {col: 'y'})
            self.df_list.append(aux)
            self.df_dict[col] = aux

        return self.df_dict

    def split_ts(ts, n):
        k, m = divmod(len(ts), n)
        return list(ts[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    def missing_rate_by_period(df, n_periods=3):
        """ Check missing rates of a time series by evenly splitted periods.
        Arg:
            df: a time series pd.DataFrame, whose index should be DatetimeIndex
            n_periods: the number of periods to be splitted into
        
        Return:
            missing_rates: a list of missing rates of each period
        """
        missing_rates = list()
        
        ts_list = split_ts(n_periods)

        for ts in ts_list:
            rate = round(df.loc[ts,].iloc[:,1].isnull().sum() / df.shape[0], 2)
            missing_rates.append(rate)
        
        return missing_rates

    def completeness_by_period(df, n_periods=3, threshold=0.9):
        """ Calculate completeness of a time series by period
        Arg:
            df: a time series pd.DataFrame, whose index should be DatetimeIndex
            n_periods: the number of periods to be splitted into
            threshold: a threshold to measure completeness, default 0.9
        Return:
            completeness_list: a binray list of completeness of each period    
        """
        completeness_list = list()
        
        missing_rates = missing_rate_by_period(n_periods)
        
        for rate in missing_rates:
            completeness = 1 - rate
            complete = 1 if completeness>=threshold else 0
            completeness_list.append(complete)

        return completeness_list

    def divide_ts(self, n_periods=3, threshold=0.9):

        self.missing_dict = dict()
        self.complete_dict = dict()

        self.ts_cat_dict = dict()

        self.comp_f_list = list()
        self.comp_h_list = list()
        self.imcomp_f_list = list()
        self.imcomp_h_list = list()

        self.ts_cat_dict['comp_full'] = self.comp_f_list
        self.ts_cat_dict['comp_half'] = self.comp_h_list
        self.ts_cat_dict['imcomp_full'] = self.imcomp_f_list
        self.ts_cat_dict['imcomp_half'] = self.imcomp_h_list

        for k, v in self.df_dict.items():
            m = self.missing_rate_by_period(v, n_periods)
            c = self.completeness_by_period(v, n_periods, threshold)

            self.missing_dict[k] = m
            self.complete_dict[k] = c
            
            if sum(c) == n_periods:
                self.comp_f_list.append(k)
            elif c[-1] == 1:
                self.comp_h_list.append(k)
            elif sum(c) == 0:
                self.imcomp_f_list.append(k)
            else:
                self.imcomp_h_list.append(k)

        return self.ts_cat_dict


    def plot_missing_by_cat(self):
        import missingno as msno

        try:
            print(self.ts_cat_dict.keys())
        except Exception as e:
            print(e)
            print('Divide the list of time series into 3 periods with threshold of completeness 0.9.')
            self.ts_cat_dict = self.divide_ts()

        for k, v in ts_cat_dict.items():
            msno.matrix(df[v], fontsize=10)
            plt.title(k, fontsize=16)

        return None
                
















