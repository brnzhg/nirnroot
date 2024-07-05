import pathlib
from pathlib import Path
import csv
import pickle
from dataclasses import dataclass
from dataclasses_json import dataclass_json

import pandas as pd

from typing import List, Optional, Tuple, Protocol, Any

from ..models.domain import RawDataMetadata, TrainingSetMetadata, ModelMetadata
from ..models.training_set import RawTrackerSourceProto, TrainingSourceProto, ModelSourceProto, TrainingSetBuilder
from ..models.model import ModelGenProto, ModelGenFactoryProto, ModelTrainerProto

from ..models import sm_discrete_kde as smkde


class ModelEnv:
    model_dir: Path


@dataclass
class ModelSource(ModelSourceProto):
    env: ModelEnv
    trainingSetBuilder: TrainingSetBuilder
    
    def get_metadata(self, id: str) -> ModelMetadata:
        for x in self._get_metadatas_iter():
            if x.id == id:
                return x
        raise FileNotFoundError(id)
    
    def get_metadatas(self) -> List[ModelMetadata]:
        return list(self._get_metadatas_iter())

    def get_trainer(self, id: str) -> ModelTrainerProto:
        md = self.get_metadata(id)
        if md.model_type == 'smkde_alt_2_1':
            down_trainer: smkde.DiscreteCondOneDimTrainer = \
                smkde.DiscreteCondOneDimTrainer(100, (2, 4))
            up_trainer: smkde.DiscreteCondOneDimTrainer = \
                smkde.DiscreteCondOneDimTrainer(100, (2, 4))
            return smkde.ModelTrainer(smkde.Trainer(down_trainer, up_trainer, .5, 1.5))
        else:
            raise Exception('Model type unknown: ' + md.model_type)

    def get_model(self, id: str) -> ModelGenFactoryProto:
        found_model: Optional[ModelGenFactoryProto]
        found_model_dir: Path
        for x in self.env.model_dir.iterdir():
           md = self._read_metadata_path(x)
           if md.id == id:
               found_model = self._read_model_path(md, x)
               found_model_dir = x
               if found_model:
                   return found_model
               break
        else:
            raise FileNotFoundError(id)
        
        trainer = self.get_trainer(id)
        df = self.trainingSetBuilder.build_training_set(id)
        model = trainer.fit(df.to_numpy())
        self._write_model_path(model, found_model_dir)
        return model
    
    def _get_metadatas_iter(self):
        for x in self.env.model_dir.iterdir():
           yield self._read_metadata_path(x)

    def _read_metadata_path(self, p: Path) -> ModelMetadata:
        with open(p / 'model-metadata.json', 'r') as f:
            return TrainingSetMetadata.from_json(f.read()) #type: ignore

    def _read_model_path(self, md: ModelMetadata, p: Path) -> Optional[ModelGenFactoryProto]:
        if not (p / 'modelgen.pkl').exists():
            return None
        with open(p / 'modelgen.pkl', 'rb') as f:
            return pickle.load(f)

    def _write_model_path(self, model, p: Path) -> None:
        with open(p / 'modelgen.pkl', 'wb') as f:
            return pickle.dump(model, f)
           
           
#with open(trainer_path, 'wb') as f:
#   pickle.dump(trainer, f) 
#
#with open(model_path, 'rb') as f:
#   loaded_model = pickle.load(f) 
#   print('sup')