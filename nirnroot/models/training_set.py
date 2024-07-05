from enum import Enum, IntEnum
from dataclasses import dataclass
from dataclasses_json import dataclass_json

import datetime
from typing import List, Optional, Tuple, Protocol, Any

import pandas as pd
from numpy.typing import NDArray

from domain import RawDataMetadata, TrainingSetMetadata, ModelMetadata
from model import ModelGenProto, ModelGenFactoryProto, ModelTrainerProto


# TODO who handles different kinds of training sets? different builders? maybe builder is wrong here or imp of proto
# abstractingboth place of data, dataset types / structure
# raw has cleaning, training has assembl

cols = ['timestamp', 'down', 'up']
#raw_data_cols = ['event', 'ts']

# TODO this handles different kinds of raw files
class RawTrackerSourceProto(Protocol):

    def get_metadata(self, id: str) -> RawDataMetadata:
        ...

    def get_metadatas(self) -> List[RawDataMetadata]:
        ...

    # should have cols, validate
    def get_data_df(self, id: str) -> pd.DataFrame:
        ...


class TrainingSourceProto(Protocol):

    def get_metadata(self, id: str) -> TrainingSetMetadata:
        ...
    
    def get_metadatas(self) -> List[TrainingSetMetadata]:
        ...



class ModelSourceProto(Protocol):
    
    def get_metadata(self, id: str) -> ModelMetadata:
        ...
    
    def get_metadatas(self) -> List[ModelMetadata]:
        ...

    def get_trainer(self, id: str) -> ModelTrainerProto:
        ...
        
    def get_model(self, id: str) -> ModelGenFactoryProto:
        ...

    
@dataclass    
class TrainingSetBuilder:
    training_source: TrainingSourceProto
    raw_source: RawTrackerSourceProto
    
    def build_training_set(self, id: str) -> pd.DataFrame:
        tmd = self.training_source.get_metadata(id)
        dfs = (self.raw_source.get_data_df(rid) for rid in tmd.raw_source_ids)
        df = pd.concat(dfs, ignore_index=True)
        return df.drop(df.columns[[0]], axis=1)







    
    

#def df_from_training_files(data_dir: pathlib.Path, fnames: List[str]):
#    col_names = ['timestamp', 'down', 'up']
#    df_from_each_file = (pd.read_csv(data_dir / 'training' / fname, names=col_names, header=None) for fname in fnames)
#    df = pd.concat(df_from_each_file, ignore_index=True)
#    #df = list(df_from_each_file)[0]
#    return df.drop(df.columns[[0]], axis=1)
