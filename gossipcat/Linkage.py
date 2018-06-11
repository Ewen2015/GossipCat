#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wang.enqun@outlook.com
license:    Apache License 2.0
"""
import time
import itertools
import pandas as pd 
import numpy as np 
import recordlinkage
from multiprocessing import Pool
import warnings
warnings.filterwarinings('ignore')

class Linkage(object):
    """docstring for Linkage"""
    def __init__(self, data, name, n_sample=50000, bin_size=500, method='jarowinkler', threshold=0.93):
        super(Linkage, self).__init__()
        self.data = data
        self.n_sample = n_sample
        self.bin_size = bin_size
        self.name = name
        self.method = method
        self.threshold = threshold
        
        self.sample = data.sample(n=self.n_sample, replace=False, random_state=0)
        self.cl = recordlinkage.FullIndex()
        self.compare_cl = recordlinkage.Compare(n_jobs=4)
        self.compare_cl.string(self.name, self.name, method=self.method, threshold=self.threshold, label=self.name)

        self.List = list(itertools.combinations(np.split(self.sample, indices_or_sections=round(self.sample.shape[0]/self.bin_size, 0)), 2))
        self.results_df = pd.DataFrame(columns=['pairs', 'company_1', 'company_2'])
        self.results_tmp = None

    def Compute(self, subsets):
        pairs_subset = self.cl.index(subsets[0], subsets[1])
        features = self.compare_cl.compute(pairs_subset, self.data)

        results = features[features[features.name]==1]
        self.results_tmp = pd.DataFrame(columns=['pairs', 'company_1', 'company_2'], index=range(results.shape[0]))

        for ind, val in enumerate(results.index):
            self.results_tmp['pairs'][ind] = val
            self.results_tmp['company_1'][ind] = self.data[self.name].ilco[val[0]]
            self.results_tmp['company_2'][ind] = self.data[self.name].ilco[val[1]]
        self.results_tmp = self.results_tmp.dropna()
        if not self.results_tmp.empty:
            print(self.results_tmp.head().to_string(index=False, header=False))
        return self.results_tmp

    def multiProcess(self, record_file=None, out_file=None):
        print('\nn_sample: '+str(self.n_sample)+
              '\tbin_size: '+str(self.bin_size)+
              '\tmethod: '+str(self.method)+
              '\tthreshold: '+str(self.threshold))

        start = time.time()
        pool = Pool()
        results = pool.map(self.Compute, self.List)
        pool.close()
        pool.join()
        duration = round((time.time()-start)/3600, 2)

        print('\ntime spend: '+str(duration)+' hours')
        if not record_file==None:
            with open(record_file, 'a') as file:
                file.write('\nn_sample: '+str(self.n_sample)+
                           '\tbin_size: '+str(self.bin_size)+
                           '\tmethod: '+str(self.method)+
                           '\tthreshold: '+str(self.threshold)+
                           '\ntime spend: '+str(duration)+' hours')

        for i in results:
            if not i.empty:
                self.results_df = pd.concat([self.results_df, i])

        if not out_file==None:
            self.results_df.to_csv(out_file, index=False)
        return self.results_df