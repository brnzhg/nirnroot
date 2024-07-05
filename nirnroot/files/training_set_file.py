import pathlib
from pathlib import Path
import csv
from dataclasses import dataclass
from dataclasses_json import dataclass_json

import pandas as pd

from typing import List, Optional, Tuple, Protocol, Any

from ..models.domain import RawDataMetadata, TrainingSetMetadata, ModelMetadata
from ..models.training_set import RawTrackerSourceProto, TrainingSourceProto, ModelSourceProto
from ..models.model import ModelGenProto, ModelGenFactoryProto, ModelTrainerProto


raw_data_cols = ['event', 'ts']

class TrainingEnv:
    raw_dir: Path
    training_dir: Path

@dataclass
class RawTrackerSource(RawTrackerSourceProto):
    
    env: TrainingEnv

    def get_metadata(self, id: str) -> RawDataMetadata:
        for x in self._get_metadatas_iter():
            if x.id == id:
                return x
        raise FileNotFoundError(id)

    def get_metadatas(self) -> List[RawDataMetadata]:
        return list(self._get_metadatas_iter())

    def get_data_df(self, id: str) -> pd.DataFrame:
        for x in self.env.raw_dir.iterdir():
           md = self._read_metadata_path(x)
           if md.id == id:
               return self._read_data_path(x)
        raise FileNotFoundError(id)

    def _get_metadatas_iter(self):
        for x in self.env.raw_dir.iterdir():
           yield self._read_metadata_path(x)

    def _read_metadata_path(self, p: Path) -> RawDataMetadata:
        with open(p / 'dataset-metadata.json', 'r') as f:
            return RawDataMetadata.from_json(f.read()) #type: ignore

    def _read_data_path(self, p: Path) -> pd.DataFrame:
        return pd.read_csv(p / 'data.csv', names=raw_data_cols, header=None)
    
    #def __init__(self, env: TrainingEnv):
    #    self.env = env

class TrainingSource(TrainingSourceProto):
    env: TrainingEnv

    def get_metadata(self, id: str) -> TrainingSetMetadata:
        for x in self._get_metadatas_iter():
            if x.id == id:
                return x
        raise FileNotFoundError(id)
    
    def get_metadatas(self) -> List[TrainingSetMetadata]:
        return list(self._get_metadatas_iter())

    def _get_metadatas_iter(self):
        for x in self.env.training_dir.iterdir():
           yield self._read_metadata_path(x)

    def _read_metadata_path(self, p: Path) -> TrainingSetMetadata:
        with open(p / 'dataset-metadata.json', 'r') as f:
            return TrainingSetMetadata.from_json(f.read()) #type: ignore

