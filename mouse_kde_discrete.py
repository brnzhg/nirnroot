import numpy as np
from scipy.optimize import brentq, RootResults
from sklearn.preprocessing import RobustScaler, PowerTransformer, MinMaxScaler
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional

import itertools
import pandas as pd
import pathlib
import csv
from typing import List, Tuple



class DiscreteCondOneDimHist:
    
    def sample(self, indep_vars) -> float:
        #TODO have to convert indep_vars into buckets
        indep_idxs = tuple(self._lookup_1d_indep_bucket(i, v) 
                           for i, v in enumerate(indep_vars))

        dep_bin_weights = self.hist[indep_idxs]

        #TODO can optimize
        chosen_dep_bin_idx: int = \
            np.random.choice(self.arange(len(dep_bin_weights)), 
                             p=dep_bin_weights)
        
        return np.random.uniform(low=self.dep_bin_edges[chosen_dep_bin_idx], 
                                 high=self.dep_bin_edges[chosen_dep_bin_idx + 1])


    def _lookup_1d_indep_bucket(self, indep_col_idx, indep_var):
        bin_count = self.indep_bins_counts[indep_col_idx]
        idx = np.digitize(indep_var, self.indep_bin_edges[indep_col_idx], right=True) - 1
        if idx == bin_count:
            return idx - 1
        return idx
    
    # returns bin_edge array (for sampling to find buckets)
    #   and digitized column
    def _make_1d_indep_buckets(self, indep_col_idx):
        indep_col = self.indep_data[:, indep_col_idx]
        bin_count = self.indep_bins_counts[indep_col_idx]

        percentiles = np.linspace(0, 100, bin_count+1)
        # bin_count + 1 edges
        bin_edges = np.percentile(indep_col, percentiles, right=True)
        dig = np.digitize(indep_col, bin_edges, right=True) - 1
        dig[dig == bin_count] = bin_count - 1

        return bin_edges, dig

    def _make_indep_buckets_and_dig_data(self):

        indep_bin_edges = []
        digs = []
        for i in range(self.num_indep_vars):
            bin_edges, dig = self._make_1d_indep_buckets(i)

            indep_bin_edges.append(bin_edges)
            digs.append(dig)

        # append digs as column vectors to make digitized data
        indep_dig_data = np.concatenate(digs, axis=1)

        return indep_bin_edges, indep_dig_data

    def _make_hist(self):
        
        hist_l = list(self.indep_bins_counts)
        hist_l.append(self.dep_bins_count)
        hist_shape = tuple(hist_l)
        hist = np.zeros(hist_shape)

        for idxs in itertools.product((range(bncnt) for bncnt in self.indep_bins_counts)):
            #TODO could optimize probably, idk if reshape is needed
            curr_indep = np.array(list(idxs))
            cdf_at_edges = np.array([self.ckde.cdf(np.array(y).reshape(-1,1), 
                                          curr_indep.reshape(-1,1)) for y in self.dep_bin_edges])
            cdf_diffs = np.ediff1d(cdf_at_edges)
            cdf_diffs_sum = np.sum(cdf_diffs)
            
            for i in range(self.dep_bins_count):
                hist[idxs][i] = cdf_diffs[i] / cdf_diffs_sum
        return hist
            


    # dep_data - 1d numpy array (obs)
    # indep_data - (obs, dims)
    # dep_bins_count - bins for sampling distr
    # indep_bins_count - (dim1 bins, dim2 bins, ...)
    def __init__(self,
                 dep_data,
                 indep_data,
                 dep_bins_count,
                 indep_bins_counts,
                 sample_min: float,
                 sample_max: float):
        self.dep_data = dep_data
        self.indep_data = indep_data
        self.dep_bins_count = dep_bins_count
        self.indep_bins_counts = indep_bins_counts
        self.sample_min = sample_min
        self.sample_max = sample_max

        self.num_indep_vars = indep_data.shape[1]

        self.indep_bin_edges, self.indep_dig_data = \
            self._make_indep_buckets_and_dig_data()

        self.ckde = KDEMultivariateConditional(dep_data, 
                                               self.indep_dig_data,
                                               dep_type='c',
                                               indep_type=('o' * self.num_indep_vars),
                                               bw='normal_reference')
                                            
        self.dep_bin_edges, self.dep_bin_width = np.linspace(sample_min, sample_max, num=dep_bins_count+1, retstep=True)

        self.hist = self._make_hist()
        
        
        
        

def df_from_training_files(data_dir: pathlib.Path, fnames: List[str]):
    col_names = ['timestamp', 'down', 'up']
    df_from_each_file = (pd.read_csv(data_dir / 'training' / fname, names=col_names, header=None) for fname in fnames)
    df = pd.concat(df_from_each_file, ignore_index=True)
    #df = list(df_from_each_file)[0]
    return df.drop(df.columns[[0]], axis=1)

def df_add_lags(df, num_lags: int):
    # Create a list to hold the original DataFrame and the lagged DataFrames
    dfs = [df if i == 0 else df.shift(i).rename(columns=lambda x: f'{x}_lag{i}') for i in range(0, num_lags + 1)]
    
    # Concatenate all the DataFrames along the columns
    df_with_lags = pd.concat(dfs, axis=1)

    return df_with_lags[num_lags:]



output_filename: str = 'bz_constant_kde_1lag.csv'

input_filenames1: List[str] = \
    ['bz_constant_050124.csv',
     'bz_constant_052324.csv',
     'bz_constant_052324_2.csv',
    ]

input_filenames2: List[str] = \
    ['raw_mouse_events_70Alchs_Focused_cleaned.csv',
     'raw_mouse_events_70Alchs_Focused2_cleaned.csv',
     'raw_mouse_events_70Alchs_Focused3_cleaned.csv',
    ]
input_filenames = input_filenames1

num_lags: int = 1
num_gen: int = 10 #input_df.shape[0]
#-------------------

# Data Transformed
data_dir: pathlib.Path = pathlib.Path.cwd() / 'tempdata'

output_filepath: pathlib.Path = data_dir / 'generated' / output_filename
input_df = df_from_training_files(
    data_dir,
    input_filenames)
data_pre_transform = input_df.to_numpy()


transformer = PowerTransformer(method='box-cox')

data = transformer.fit_transform(data_pre_transform)
data_df = pd.DataFrame(data, columns=input_df.columns)

data_df_with_lags = df_add_lags(data_df, num_lags)
data_with_lags = data_df_with_lags.to_numpy()

print('last obs')
print(data_with_lags[-5:,:])

# data has is numpy array with lags
#----------------------------------
# prepping model 
down_data = data[:,0].reshape(-1,1)
up_data = data[:,1].reshape(-1,1)
lag_data = data_with_lags[:,2:]

down_indep = lag_data
up_indep = np.hstack([down_data, lag_data])

max_obs_pre_transform = \
    1.5 * np.array([np.max(data_pre_transform[:,0]), 
                    np.max(data_pre_transform[:,1])])
min_obs_pre_transform = \
    .5 * np.array([np.min(data_pre_transform[:,0]), 
                   np.min(data_pre_transform[:,1])])

sample_max = transformer.transform([max_obs_pre_transform])[0]
sample_min = transformer.transform([min_obs_pre_transform])[0]

# reverse matrix? nah

# for debugging
transformed_min = (np.min(data[:,0]), np.min(data[:,1]))

#-----------------
# train model
down_hist = DiscreteCondOneDimHist(down_data,
                                   down_indep, 
                                   dep_bins_count=200,
                                   indep_bins_counts=(10, 10),
                                   sample_min=sample_min,
                                   sample_max=sample_max)

up_hist = DiscreteCondOneDimHist(up_data,
                                 up_indep,
                                 dep_bins_count=200,
                                 indep_bins_counts=(10, 10, 10),
                                 sample_min=sample_min,
                                 sample_max=sample_max)


gen_data = []

last_lags = np.concatenate([data[-1], lag_data[-1, :-2]])
print(f'last lags seed: {last_lags}')

# TODO can optimize by just repopulating array
for i in range(num_gen):

    down_sample = down_hist.sample(last_lags)

    down_and_last_lags = np.concatenate([np.array([down_sample]), last_lags])
    up_sample = up_hist.sample(down_and_last_lags)

    sample = [down_sample, up_sample]
    last_lags = np.concatenate([np.array(sample), last_lags[:-2]])

    gen_data.append(sample)

    last_lags = np.concatenate([sample, last_lags[:-2]])


unscaled_gen_data = transformer.invert(np.array(gen_data))

print('worked?')
print(unscaled_gen_data[:5])