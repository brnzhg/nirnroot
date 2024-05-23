import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

def plot_binary_stripes():
    pass


def plot_hists():
    pass



clean_raw_data_df = pd.read_csv('raw_mouse_events_cleaned.csv', header=None)
learned_data_df = pd.read_csv('learned_events.csv', header=None)

clean_raw_data = \
    clean_raw_data_df.drop(clean_raw_data_df.columns[[0]], axis=1).to_numpy()
learned_data= learned_data_df.to_numpy()

print(clean_raw_data[:5])
print(learned_data[:5])

clean_raw_data_bin = create_binary_from_click_duration_array(clean_raw_data)
learned_data_bin = create_binary_from_click_duration_array(learned_data)

print(clean_raw_data_bin[:5])
print(learned_data_bin[:5])



clean_raw_ts = clean_raw_data_bin[:, 0]
clean_raw_bs = clean_raw_data_bin[:, 1]
learned_ts = learned_data_bin[:, 0]
learned_bs = learned_data_bin[:, 1]

fig, axes = plt.subplots(2, 1)

axes[0].set_title('Real')
axes[0].vlines(clean_raw_ts, ymin=0, ymax=1, color='black', linewidth=.2)
axes[0].fill_between(clean_raw_ts, clean_raw_bs, 0, step='post', color='blue') #where=(clean_raw_bs == 1))

axes[1].set_title('Gen')
axes[1].vlines(learned_ts, ymin=0, ymax=1, color='black', linewidth=.2)
axes[1].fill_between(learned_ts, clean_raw_bs, 0, step='post', color='blue') #where=(clean_raw_bs == 1))

# Create a step plot
#plt.vlines(clean_raw_ts, ymin=0, ymax=1, color='black', linewidth=.2)
#plt.step(clean_raw_ts, clean_raw_bs, where='post', label='Binary Value', color='black')
# Fill the intervals with different colors

#plt.fill_between(clean_raw_ts, clean_raw_bs, 0, step='post', color='blue') #where=(clean_raw_bs == 1))
#plt.fill_between(clean_raw_ts, clean_raw_bs, 1, step='post', color='blue') #where=(clean_raw_bs == 1))
#plt.fill_between(clean_raw_ts, clean_raw_bs, step='post', where=(clean_raw_bs == 1), color='lightblue', alpha=0.6, interpolate=True, label='1')
#plt.fill_between(clean_raw_ts, clean_raw_bs, step='post', where=(clean_raw_bs == 0), color='lightcoral', alpha=0.6, interpolate=True, label='0')

# Customize the plot
#plt.grid(True)
#plt.legend()

# Show the plot
plt.show()
