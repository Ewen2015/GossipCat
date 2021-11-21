#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import time
import itertools
import pandas as pd 
import numpy as np 
import recordlinkage
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

class Linkage(object):
    """docstring for Linkage"""
    def __init__(self, data, new_data, name, n_sample=50000, bin_size=500, block=None, method='jarowinkler', threshold=0.93):
        super(Linkage, self).__init__()
        self.data = data
        self.new_data = pd.DataFrame(new_data)
        self.n_sample = n_sample
        self.bin_size = bin_size
        self.name = name
        self.block = block
        self.method = method
        self.threshold = threshold
        
        self.sample = data.sample(n=self.n_sample, replace=False, random_state=0)

        if self.block != None:
            self.indexer = recordlinkage.BlockIndex(on=self.block)
        else:
            self.indexer = recordlinkage.FullIndex()
        self.compare_cl = recordlinkage.Compare(n_jobs=4)
        self.compare_cl.string(self.name, self.name, method=self.method, threshold=self.threshold, label=self.name)

        if self.new_data.empty:
            self.List = list(itertools.combinations(np.split(self.sample, indices_or_sections=round(self.n_sample/self.bin_size, 0)), 2))
        else:
            self.Linst = list(np.split(self.sample, indices_or_sections=round(self.n_sample/self.bin_size, 0)))
        self.results_df = pd.DataFrame(columns=['pairs', 'company_1', 'company_2'])
        self.results_tmp = None

    def Compute(self, subsets):
        if self.new_data.empty:
            pairs_subset = self.indexer.index(subsets[0], subsets[1])
            features = self.compare_cl.compute(pairs_subset, subsets[0], subsets[1])
        else:
            pairs_subset = self.indexer.index(subsets, self.new_data)
            features = self.compare_cl.compute(pairs_subset, self.sample, self.new_data)

        results = features[features[self.name]==1]
        self.results_tmp = pd.DataFrame(columns=['pairs', 'company_1', 'company_2'], index=range(results.shape[0]))

        for ind, val in enumerate(results.index):
            self.results_tmp['pairs'][ind] = val
            self.results_tmp['company_1'][ind] = self.data[self.name].iloc[val[0]]
            self.results_tmp['company_2'][ind] = self.data[self.name].iloc[val[1]]
        self.results_tmp = self.results_tmp.dropna()
        self.results_tmp = self.results_tmp[self.results_tmp['company_1']!=self.results_tmp['company_2']]
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

        for i in results:
            if not i.empty:
                self.results_df = pd.concat([self.results_df, i])

        if not record_file==None:
            with open(record_file, 'a') as file:
                file.write('\nn_sample: '+str(self.n_sample)+
                           '\tbin_size: '+str(self.bin_size)+
                           '\tmethod: '+str(self.method)+
                           '\tthreshold: '+str(self.threshold)+
                           '\ntime spend: '+str(duration)+' hours'+
                           '\tnumber_of_records: '+str(self.results_df.shape[0]))

        if not out_file==None:
            self.results_df.to_csv(out_file, index=False)
        return self.results_df