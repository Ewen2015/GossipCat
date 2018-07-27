import time
import gc
import itertools
import pandas as pd 
import numpy as  np 
import jellyfish.cjellyfish as jelly 
from multiprocessing import Pool 


class Similarity(object):
    """docstring for Similarity"""
    def __init__(self, data_file, column_name, threshold=0.93, bin_size=600, results_file='results'):
        super(Similarity, self).__init__()
        self.data_file = data_file
        self.column_name = column_name
        self.threshold = threshold
        self.bin_size = bin_size
        self.results_file = results_file
        
        self.data = pd.read_csv(self.data_file)
        self.data = self.data[self.column_name]

        self.tail_size = self.data.shape[0]%bin_size
        self.data_tail = self.data.tail(self.tail_size)
        self.data_fixed = self.data.iloc[self.tail_size:]
        self.bin_list = np.split(self.data_fixed, indices_or_sections=(self.data.shape[0]-self.tail_size)/self.bin_size)
        self.bin_list.append(self.data_tail)
        self.com_list = list(itertools.combinations(self.bin_list, 2))

    def _map(self, line):
        sim = jelly.jaro_winkler(line[0], line[1])
        return [line[0], line[1], '%.3f' % sim]

    def _foo(self, List):
        results = list(filter(lambda line: float(line[2]>self.threshold, 
                              list(map(lambda line: self._map(line), 
                                       list(itertools.product(List[0], List[1])))))))
        with open(self.results_file, 'a') as file:
            for item in results:
                file.write(str(item)+'\n')
        del results
        gc.collect()
        return None

    def _foo1(self, List):
        results = list(filter(lambda line: float(line[2]>self.threshold, 
                              list(map(lambda line: self._map(line), 
                                       list(itertools.combinations(List, 2)))))))
        with open(self.results_file, 'a') as file:
            for item in results:
                file.write(str(item)+'\n')
        del results
        gc.collect()
        return None        
        
    def Run(self):
        print('='*10, 'configN', '='*10)
        print('data_file', '\t', self.data_file)
        print('results_file', '\t', self.results_file)
        print('column_name', '\t', self.column_name)
        print('threshold', '\t', self.threshold)
        print('bin_size', '\t', self.bin_size)

        print('='*10, 'stage 1', '='*10)
        start = time.time()
        pool = Pool()
        pool.map(self._foo, self.com_list)
        pool.close()
        time1 = round((time.time()-start)/60, 2)
        print('time spend', time1, 'min')

        print('='*10, 'stage 2', '='*10)
        start = time.time()
        pool = Pool()
        pool.map(self._foo1, self.bin_list)
        pool.close()
        time2 = round((time.time()-start)/60, 2)
        print('time spend', time2, 'min')
        
        print('='*10, 'summary', '='*10)
        print('total time spend', time1+time2, 'min')
        return None



        