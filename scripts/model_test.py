import numpy as np
# from sklearn.preprocessing import RobustScaler, PowerTransformer, MinMaxScaler
# from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional

from dataclasses import dataclass
import itertools
import pandas as pd
import pathlib
import csv
import pickle
import sys

from typing import List, Tuple

sys.path.append('../nirnroot')
from nirnroot.models import model as mdl
from nirnroot.models import sm_discrete_kde as smkde


def df_from_training_files(data_dir: pathlib.Path, fnames: List[str]):
    col_names = ['timestamp', 'down', 'up']
    df_from_each_file = (pd.read_csv(data_dir / 'training' / fname, names=col_names, header=None) for fname in fnames)
    df = pd.concat(df_from_each_file, ignore_index=True)
    #df = list(df_from_each_file)[0]
    return df.drop(df.columns[[0]], axis=1)

    
    
output_filename: str = 'constant_smkde_sym2.csv'

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
    
#data_dir: pathlib.Path = pathlib.Path.cwd().parent / 'tempdata'
data_dir: pathlib.Path = pathlib.Path.cwd() / 'tempdata'

output_filepath: pathlib.Path = data_dir / 'generated' / output_filename
input_df = df_from_training_files(
    data_dir,
    input_filenames)
data = input_df.to_numpy()

down_trainer: smkde.DiscreteCondOneDimTrainer = \
    smkde.DiscreteCondOneDimTrainer(100, (2, 4))
up_trainer: smkde.DiscreteCondOneDimTrainer = \
    smkde.DiscreteCondOneDimTrainer(100, (2, 4))

trainer = smkde.Trainer(down_trainer, up_trainer, .5, 1.5)

model: smkde.SMDiscreteJointKde = trainer.fit(data)

model_gen = smkde.SMDiscreteJointKdeGen(model, data, data.shape[0] - 1)

genned = model_gen.gen_next_n(2400)

with open(output_filepath, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(genned)
    

model_path = data_dir / 'modelpickle.pkl'
trainer_path = data_dir / 'trainerpickle.pkl'

#with open(model_path, 'wb') as f:
#   pickle.dump(model, f) 
#
#with open(trainer_path, 'wb') as f:
#   pickle.dump(trainer, f) 
#
#with open(model_path, 'rb') as f:
#   loaded_model = pickle.load(f) 
#   print('sup')
#
#with open(trainer_path, 'rb') as f:
#   loaded_trainer = pickle.load(f) 
#   print('sup')
