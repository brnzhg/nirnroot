import numpy as np
from scipy.optimize import brentq
#from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import RobustScaler, PowerTransformer, MinMaxScaler
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional

import pandas as pd
import pathlib
import csv
from typing import List

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

# Handle outliers with RobustScaler
#scaler = RobustScaler()
#data_scaled = scaler.fit_transform(data)

# Handle skewness with PowerTransformer (Yeo-Johnson by default)
#transformer = PowerTransformer(method='yeo-johnson')
#data_transformed = transformer.fit_transform(data_scaled)

output_filename: str = '70Alchs_Focused_kde_2lag.csv'

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

num_lags: int = 2
num_gen: int = 4000 #input_df.shape[0]
#-------------------

data_dir: pathlib.Path = pathlib.Path.cwd() / 'tempdata'

output_filepath: pathlib.Path = data_dir / 'generated' / output_filename
input_df = df_from_training_files(
    data_dir,
    input_filenames)

transformer = PowerTransformer(method='box-cox')
input_transformed = transformer.fit_transform(input_df.to_numpy())
#input_transformed = np.log(input_df.to_numpy())

input_df_transformed = pd.DataFrame(input_transformed, columns=input_df.columns)
df = df_add_lags(input_df_transformed, num_lags)

data = df.to_numpy()

print('last obs')
print(data[-5:,:])

def cdf_conditional(kde, x, y_target):
    # cdf of kde conditional on x, evaluated at y_target
    return kde.cdf(np.array(y_target).reshape((-1,1)), np.array(x).reshape((-1,1)))

# inverse-transform-sampling
def sample_conditional_single(kde, x):
    # sample conditional on x
    u = np.random.random()
    # 1-d root-finding
    def func(y):
        return cdf_conditional(kde, x, y) - u
    sample_y = brentq(func, -99999999, 99999999)  # read brentq-docs about these constants
                                                # constants need to be sign-changing for the function
    return sample_y

def make_ckde_helper(dep_var, indep_var):
    #bw='normal_reference'
    return KDEMultivariateConditional(dep_var, indep_var, 'c', ('c' * indep_var.shape[1]), bw='normal_reference')

down_data = data[:,0].reshape(-1,1)
up_data = data[:,1].reshape(-1,1)
lag_data = data[:,2:]
down_indep = lag_data
up_indep = np.hstack([down_data, lag_data])


print(f'down indep: {down_indep[-1]}')
print(f'up indep: {up_indep[-1]}')


down_ckde = make_ckde_helper(down_data, down_indep)
up_ckde = make_ckde_helper(up_data, up_indep) 


gen_data = []
# seed x with last training data observations, drop last 2 cols of lags
last_lags = np.concatenate([data[-1, :2], lag_data[-1, :-2]])
print(f'last lags seed: {last_lags}')

for i in range(num_gen):
    down_sample = sample_conditional_single(down_ckde, last_lags)

    down_and_last_lags = np.concatenate([np.array([down_sample]), last_lags])
    
    up_sample = sample_conditional_single(up_ckde, down_and_last_lags)

    # drop last 2 cols of lags
    last_lags = np.concatenate([np.array([down_sample, up_sample]), last_lags[:-2]])
    #x = np.concatenate([y, x[:-2]])

    gen_data.append([down_sample, up_sample])
    
unscaled_gen_data = transformer.inverse_transform(np.array(gen_data))
#unscaled_gen_data = np.exp(np.array(gen_data))

print('worked?')
print(unscaled_gen_data[:5])

with open(output_filepath, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(unscaled_gen_data)
    #for generated_row in gen_data:
    #    writer.writerow(list(generated_row))
# ideas - normalize by 70 and denormalize, by random variance and mean




# have x starting array, generate y, then make updated x array
# x array can be seeded by last row of dp_data plus indep_data - 2






