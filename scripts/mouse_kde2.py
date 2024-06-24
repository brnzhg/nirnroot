import numpy as np
from sklearn.preprocessing import RobustScaler, PowerTransformer, MinMaxScaler
#from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional
import KDEpy 
from KDEpy.bw_selection import silvermans_rule, improved_sheather_jones
from KDEpy.utils import autogrid


from itertools import product
import pandas as pd
import pathlib
import csv
from typing import List, Tuple

# shamelessly taken from
# https://stackoverflow.com/questions/60223573/conditional-sampling-from-multivariate-kernel-density-estimate-in-python


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


# transforms data without lags
class TransformerWithBw:


    def transform(self, arr):
        r = self.transformer.transform(arr)
        return r / self.bw

    def invert(self, arr):
        r = self.transformer.inverse_transform(arr)
        return r * self.bw
    

    def __init__(self, 
                 transformer: PowerTransformer, 
                 down_bw: float, 
                 up_bw: float):
        self.transformer = transformer
        self.down_bw = down_bw
        self.up_bw = up_bw
        self.bw = np.array([down_bw, up_bw])


def fit_transform_bw(data) -> TransformerWithBw:
    transformer = PowerTransformer(method='box-cox')
    pt_data = transformer.fit_transform(data)

    down_bw = 1 #silvermans_rule(pt_data[:, [0]])
    up_bw = 1 #silvermans_rule(pt_data[:, [1]])

    return TransformerWithBw(transformer, down_bw, up_bw)

# grid is (obs, dims)
# grid_rng is arange(0, obs)
# steps is (dim1 step, dim2 step)
# sample a row, then uniformly select within step in each dim
def grid_sample(ys, grid, grid_rng, steps):
    ys_normed = ys / np.sum(ys)
    
    idx:int = np.random.choice(grid_rng, p=ys_normed)

    sample_lbs = grid[idx]
    sample_ubs = sample_lbs + steps

    return np.random.uniform(low=sample_lbs, high=sample_ubs)

 

    



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
num_gen: int = 1000 #input_df.shape[0]
#-------------------

# Data Transformed
data_dir: pathlib.Path = pathlib.Path.cwd() / 'tempdata'

output_filepath: pathlib.Path = data_dir / 'generated' / output_filename
input_df = df_from_training_files(
    data_dir,
    input_filenames)
data_pre_transform = input_df.to_numpy()

# https://github.com/tommyod/KDEpy/issues/81
# note not reverse transforming Xs until end, should be no reason to transform probs then
transformer_bw: TransformerWithBw = fit_transform_bw(data_pre_transform)
#transformer = PowerTransformer(method='box-cox')
#input_transformed = transformer.fit_transform(data_pre_transform)

data = transformer_bw.transform(data_pre_transform)
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

max_obs_pre_transform = \
    1.5 * np.array([np.max(data_pre_transform[:,0]), 
                    np.max(data_pre_transform[:,1])])
min_obs_pre_transform = \
    .5 * np.array([np.min(data_pre_transform[:,0]), 
                   np.min(data_pre_transform[:,1])])

grid_max = transformer_bw.transform([max_obs_pre_transform])[0]
grid_min = transformer_bw.transform([min_obs_pre_transform])[0]

# for debugging
transformed_min = (np.min(data[:,0]), np.min(data[:,1]))

bins_1d: int = 20
grid_downs, grid_d_step = np.linspace(grid_min[0], grid_max[0], num=bins_1d, retstep=True)
grid_ups, grid_u_step = np.linspace(grid_min[1], grid_max[1], num=bins_1d, retstep=True)
grid = np.array(list(product(grid_downs, grid_ups)))
grid_rng = np.arange(grid.shape[0])

joint_kde = KDEpy.TreeKDE(kernel="gaussian", bw=1).fit(data_with_lags)
#---------------------------
#TODO alternate approach, fit 2 KDEs and like before
#TODO alternate approach, make giant numpy array and sample from it, wide buckets for conditioning
#TODO alternate approach, joint KDE only

# generator

gen_data = []

last_lags = np.concatenate([data[-1], lag_data[-1, :-2]])
print(f'last lags seed: {last_lags}')

# TODO can optimize by just repopulating array
for i in range(num_gen):
    eval_grid = np.hstack([np.repeat([last_lags], grid.shape[0], axis=0), grid])

    # should be 1d array
    ys = joint_kde.evaluate(eval_grid)

    sample = grid_sample(ys, grid, grid_rng, steps=(grid_d_step, grid_u_step))

    gen_data.append(sample)

    last_lags = np.concatenate([sample, last_lags[:-2]])



unscaled_gen_data = transformer_bw.invert(np.array(gen_data))

print('worked?')
print(unscaled_gen_data[:5])