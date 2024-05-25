import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from typing import List


def create_binary_from_click_duration_array(click_durr_arr):
    l = []
    t = 0
    for r in click_durr_arr:
        l.append([t, 1])
        t += r[0]
        l.append([t, 0])
        t += r[1]
    l.append([t, 1])
    return np.array(l)

def plot_binary_stripes(clean_raw_ts, clean_raw_bs, learned_ts, learned_bs):

    fig, axes = plt.subplots(2, 1)

    axes[0].set_title('Real')
    axes[0].vlines(clean_raw_ts, ymin=0, ymax=1, color='black', linewidth=.2)
    axes[0].fill_between(clean_raw_ts, clean_raw_bs, 0, step='post', color='blue') #where=(clean_raw_bs == 1))

    axes[1].set_title('Gen')
    axes[1].vlines(learned_ts, ymin=0, ymax=1, color='black', linewidth=.2)
    axes[1].fill_between(learned_ts, learned_bs, 0, step='post', color='blue') #where=(clean_raw_bs == 1))


def plot_hists(clean_raw_data, learned_data):
    fig, axes = plt.subplots(2, 2)

    n_bins = 100

    axes[0, 0].set_title('Real Down')
    #axes[0, 0].hist(clean_raw_data[:,0], bins=n_bins, range=(0, .4))
    axes[0, 0].hist(clean_raw_data[:,0], bins=n_bins)
    axes[1, 0].set_title('Gen Down')
    #axes[1, 0].hist(learned_data[:,0], bins=n_bins, range=(0, .4))
    axes[1, 0].hist(learned_data[:,0], bins=n_bins)

    axes[0, 1].set_title('Real Up')
    #axes[0, 1].hist(clean_raw_data[:,1], bins=n_bins, range=(0, .4))
    axes[0, 1].hist(clean_raw_data[:,1], bins=n_bins)
    axes[1, 1].set_title('Gen Up')
    #axes[1, 1].hist(learned_data[:,1], bins=n_bins, range=(0, .4))
    axes[1, 1].hist(learned_data[:,1], bins=n_bins)

def df_from_training_files(data_dir: pathlib.Path, fnames: List[str]):
    col_names = ['timestamp', 'down', 'up']
    df_from_each_file = (pd.read_csv(data_dir / 'training' / fname, names=col_names, header=None) for fname in fnames)
    df = pd.concat(df_from_each_file, ignore_index=True)
    #df = list(df_from_each_file)[0]
    return df.drop(df.columns[[0]], axis=1)

data_dir: pathlib.Path = pathlib.Path.cwd() / 'tempdata'


#clean_raw_data_df = pd.read_csv(data_dir / 'training' / 'raw_mouse_events_70Alchs_Focused_cleaned.csv', header=None)
#clean_raw_data_df = clean_raw_data_df.drop(clean_raw_data_df.columns[[0]], axis=1)
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
input_filenames = input_filenames2
clean_raw_data_df = df_from_training_files(
    data_dir,
    input_filenames)
#learned_data_df = pd.read_csv(data_dir / 'generated' / 'learned_events_70Alchs_Focused_minmax5.csv', header=None)
learned_data_df = pd.read_csv(data_dir / 'generated' / '70Alchs_Focused_kde_2lag.csv', header=None)

clean_raw_data = clean_raw_data_df.to_numpy()
learned_data= learned_data_df.to_numpy()

print('data')
print(clean_raw_data[:5])
print(learned_data[:5])

# sample data to view
n = 50 #clean_raw_data_bin.shape[0]
raw_idx_start = np.random.randint(0, clean_raw_data.shape[0] - n)
learned_idx_start = np.random.randint(0, learned_data.shape[0] - n)

clean_raw_data_smp = clean_raw_data[raw_idx_start:(raw_idx_start + n),:]
learned_data_smp = learned_data[learned_idx_start:(learned_idx_start + n),:]

print('sampled data')
print(clean_raw_data_smp[:5])
print(learned_data_smp[:5])
print('sampling train from ' + str(raw_idx_start))
print('sampling learned from ' + str(learned_idx_start))


clean_raw_data_bin_smp = create_binary_from_click_duration_array(clean_raw_data_smp)
learned_data_bin_smp = create_binary_from_click_duration_array(learned_data_smp)

print('sampled bin data')
print(clean_raw_data_bin_smp[:5])
print(learned_data_bin_smp[:5])

clean_raw_ts = clean_raw_data_bin_smp[:, 0]
clean_raw_bs = clean_raw_data_bin_smp[:, 1]
learned_ts = learned_data_bin_smp[:, 0]
learned_bs = learned_data_bin_smp[:, 1]

plot_binary_stripes(clean_raw_ts, clean_raw_bs, learned_ts, learned_bs)
plot_hists(clean_raw_data, learned_data)

print(clean_raw_data_df.describe())
print(learned_data_df.describe())


# Show the plot
plt.show()
