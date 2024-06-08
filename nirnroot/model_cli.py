from pathlib import Path

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from typing import List, Optional, Tuple, Protocol, Any

from models.model import ModelGenProto



@dataclass_json
@dataclass
class ModelMetadata:
    id: str
    name: str
    model_type: str
    training_id: str
    
# implementation sfor each model specifically
# this is CLI crap layer, each model logic dont need to know this, dep other way
# bridges general CLI layer and modelgen layer, it's the most specific
class ModelGenLoaderProto(Protocol):
    model_type: str # tells them what type it can load

    #TODO could take in ModelMetadata or just load itself 
    def load_model_gen(self, md: ModelMetadata, model_path: Path) -> ModelGenProto:
        ...