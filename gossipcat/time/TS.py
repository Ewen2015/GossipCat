#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import pandas as pd 

class TSDict(object):
    """docstring for TSDict"""
    def __init__(self, df, col_date):
        super(TSDict, self).__init__()
        self.df = df
        self.col_date = col_date

        self.df = self.process_date()
        self.columns = self.df.columns
        self.df_dict = self.to_df_dict()

    def process_date(self):
        self.df[self.col_date] = pd.to_datetime(self.df[self.col_date])
        self.df.index = self.df[self.col_date]
        del self.df[self.col_date]
        self.df = self.df.replace(0, None)
        return self.df
    
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

    def split_ts(self, ts, n):
        k, m = divmod(len(ts), n)
        return list(ts[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    def missing_rate_by_period(self, df, n_periods=3):
        """ Check missing rates of a time series by evenly splitted periods.
        Arg:
            df: a time series pd.DataFrame, whose index should be DatetimeIndex
            n_periods: the number of periods to be splitted into
        
        Return:
            missing_rates: a list of missing rates of each period
        """
        self.n_periods = n_periods
        
        missing_rates = list()
        
        ts_list = self.split_ts(df.index, self.n_periods)

        for ts in ts_list:
            rate = round(df.loc[ts,].iloc[:,1].isnull().sum() / df.shape[0], 2)
            missing_rates.append(rate)
        
        return missing_rates

    def completeness_by_period(self, df, n_periods=3, threshold=0.9):
        """ Calculate completeness of a time series by period
        Arg:
            df: a time series pd.DataFrame, whose index should be DatetimeIndex
            n_periods: the number of periods to be splitted into
            threshold: a threshold to measure completeness, default 0.9
        Return:
            completeness_list: a binray list of completeness of each period    
        """
        self.n_periods = n_periods
        self.threshold = threshold
        
        completeness_list = list()
        
        missing_rates = self.missing_rate_by_period(df, self.n_periods)
        
        for rate in missing_rates:
            completeness = 1 - rate
            complete = 1 if completeness>=self.threshold else 0
            completeness_list.append(complete)

        return completeness_list

    def group_ts_by_period_percentage(self, n_periods=3, threshold=0.9):
        self.n_periods = n_periods
        self.threshold = threshold
        
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
            m = self.missing_rate_by_period(v, self.n_periods)
            c = self.completeness_by_period(v, self.n_periods, self.threshold)

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


    def max_consecutive_mn(self, df, col):
        tmp = df.copy()
        if tmp[col].isnull().any():        
            tmp['Group'] = tmp[col].notnull().astype(int).cumsum()
            tmp = tmp[tmp[col].isnull()]
            tmp['count'] = tmp.groupby('Group')['Group'].transform('size')
            result = tmp.drop_duplicates('Group').sort_values('count')['count'][-1]
            return result
        else:
            return 0

    def df_missing_summary(self, col='y'):
        self.dict_cns = dict()
        self.dict_cnt = dict()
        self.dict_ptg = dict()

        for k, v in self.df_dict.items():
            self.dict_cns[k] = self.max_consecutive_mn(v, col) 
            self.dict_cnt[k] = v[col].isnull().sum()
            self.dict_ptg[k] = round(self.dict_cnt[k]/v.shape[0], 2)

        self.df_missing = pd.DataFrame({'consecutive': self.dict_cns, 
                                        'count': self.dict_cnt, 
                                        'percentage': self.dict_ptg})
        return self.df_missing

    def group_ts_by_consecutives(self, n_consecutives=6, n_tails=3):
        self.n_consecutives=n_consecutives
        self.n_tails=n_tails

        self.df_missing = self.df_missing_summary()

        self.ts_cat_dict = dict()

        self.complete = list()
        self.imcomp_active = list()
        self.imcomp_dormant = list()

        self.ts_cat_dict['complete'] = self.complete
        self.ts_cat_dict['imcomp_active'] = self.imcomp_active
        self.ts_cat_dict['imcomp_dormant'] = self.imcomp_dormant

        for k, v in self.dict_cns.items():
            if v <= self.n_consecutives:
                if self.df[k].tail(self.n_tails).isnull().any():
                    self.imcomp_dormant.append(k)
                else:
                    self.complete.append(k)
            else:
                if self.df[k].tail(self.n_tails).isnull().any():
                    self.imcomp_dormant.append(k)
                else:
                    self.imcomp_active.append(k)

        return self.ts_cat_dict


    def plot_missing_by_cat(self, by_consecutives=True):
        import missingno as msno
        import matplotlib.pyplot as plt

        try:
            print(self.ts_cat_dict.keys())
        except Exception as e:
            print(e)
            if by_consecutives:
                self.ts_cat_dict = self.group_ts_by_consecutives()
                print('Grouped the list of time series by {} consecutive missing values and {} tails.'.format(self.n_consecutives, self.n_tails))
            else:
                self.ts_cat_dict = self.group_ts_by_period_percentage()
                print('Grouped the list of time series into {} periods with threshold of completeness {}.'.format(self.n_periods, self.threshold))

        for k, v in self.ts_cat_dict.items():
            msno.matrix(self.df[v], fontsize=10)
            plt.title(k, fontsize=16)

        return None