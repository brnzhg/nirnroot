from enum import Enum, IntEnum
from dataclasses import dataclass
from dataclasses_json import dataclass_json

import datetime
from typing import List, Optional, Tuple, Protocol


class TrackerEvent(IntEnum):
    MouseUp = 0,
    MouseDown = 1,
    TrackPause = 3,
    TrackResume = 4

    
@dataclass_json
@dataclass
class RawDataMetadata:
    user: str
    label: str
    id: str
    raw_data_type: str

@dataclass_json
@dataclass
class TrainingSetMetadata:
    title: str
    user: str
    id: str
    raw_source_ids: List[str]

@dataclass_json
@dataclass
class ModelMetadata:
    id: str
    name: str
    model_type: str
    training_set_id: str