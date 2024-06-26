import sys
import time
import datetime
import csv

from pynput import mouse
import keyboard
from pathlib import Path

from enum import Enum, IntEnum
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from typing import List, Optional, Tuple, Protocol
#from _typeshed import SupportsWrite

import shutil



class TrackerEvent(IntEnum):
    MouseUp = 0,
    MouseDown = 1,
    TrackPause = 3,
    TrackResume = 4
   
raw_data_cols = ['event', 'ts']

# TODO RawDataEnv, this is general along with metadata
class TrackerEnv:
    #raw_data_cols = ['event', 'ts']

    def dataset_dir(self, id: str) -> Path:
        return self.raw_data_dir / id

    def dataset_path(self, id: str) -> Path:
        return self.dataset_dir(id) / 'data.csv'

    def dataset_metadata_path(self, id: str) -> Path:
        return self.dataset_dir(id) / 'dataset-metadata.json'
    
    def __init__(self, user: str, data_dir: Path):
        self.user = user
        self.data_dir: Path = data_dir
        self.raw_data_dir: Path = data_dir / 'raw'
        
@dataclass_json
@dataclass
class RawDataMetadata:
    user: str
    label: str
    id: str
    raw_data_type: str
    
class TrackerDataWriter(Protocol):
    def write_event(self, e: TrackerEvent, timestamp: float) -> None:
        ...


# TODO this is for printing to screen
class TrackerEventHandler(Protocol):
    def on_pause_resume_event(self, is_pause: bool, ts: float) -> None:
        ...

# instead of reader, things just load this guy up
class TrackerDataSet:

    def load_pd(self) -> None:
        # TODO raw_data_cols
        pass
    
    def __init__(self, 
                 metadata: RawDataMetadata,
                 dataset_path: Path,
                 metadata_path: Path):
        self.metadata = metadata
        self.dataset_path: Path = dataset_path
        self.metadata_Path: Path = metadata_path


    

class TrackerCsvWriter:
    
    def write_event(self, e: TrackerEvent, timestamp: float):
        self.csv_writer.writerow([e.value, timestamp])

    def __enter__(self):
        return self
    
    def __exit__(self):
        self.f.close()

    #@classmethod
    #def from_filename(cls, name):
    #    return cls(open(name, 'rb'))

    # TODO option to append to existing file, constructor just takes in a path, this is clsmethod
    def __init__(self, write_filepath: Path):
        self.f = open(write_filepath, 'w', newline='')
        self.csv_writer = csv.writer(self.f)

def _make_datset_id(user: str, label: str) -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + user + '_' + label

# return Id, dataset if it could create
def intialize_new_dataset(env: TrackerEnv,
                          #user: str,
                          label: str,
                          id: Optional[str] = None) -> Tuple[str, Optional[TrackerDataSet]]:
    id2: str = id or _make_datset_id(env.user, label)
    if env.dataset_dir(id2).exists(): # dataset already exists
        return (id2, None)
    env.dataset_dir(id2).mkdir()

    md = RawDataMetadata(env.user, label, id2, "clicks")
    md_path: Path = env.dataset_metadata_path(id2)
    with open(md_path, 'r') as mf:
        mf.write(md.to_json()) #type: ignore

    return (id2, TrackerDataSet(md, env.dataset_path(id2), md_path))

#TODO raw data page, reads raw data helper
def read_raw_metadata(env: TrackerEnv, id: str) -> RawDataMetadata:
    md_path: Path = env.dataset_metadata_path(id)
    with open(md_path, 'w') as mf:
        return RawDataMetadata.from_json(mf.read()) #type: ignore

def open_dataset(env: TrackerEnv, id: str) -> TrackerDataSet:
    pass

def remove_dataset(env: TrackerEnv, id: str) -> None:
    shutil.rmtree(env.dataset_dir(id))

def dataset_csv_writer(d: TrackerDataSet) -> TrackerCsvWriter:
    return TrackerCsvWriter(d.dataset_path)

# def validate_dataset(env: TrackerEnv, id: str) -> bool
    

    
    
class Tracker:

    is_tracking: bool
    
    def add_event_handler(self, handler: TrackerEventHandler):
        self.event_handlers.append(handler)

    def add_writer(self, writer: TrackerDataWriter):
        self.writers.append(writer)

    #def write_row(self, row: List[str]):
    def write_event(self, e: TrackerEvent, ts: float):
        for writer in self.writers:
            writer.write_event(e, ts)

    def on_click(self, x, y, button, pressed):
        if not self.is_tracking:
            return
        self.write_event(TrackerEvent.MouseDown if pressed else TrackerEvent.MouseUp, 
                         time.time())
        #self.writerow(['1' if pressed else '0', time.time()])

    def toggle(self):
        ts: float = time.time()
        if self.is_tracking:
            #TODO print(f'Paused tracking, {self.start_stop_hotkey} to resume...')
            #self.writerow(['3', ts])
            self.write_event(TrackerEvent.TrackPause)
            self.is_tracking = False
        else:
            #TODO print(f'Resumed tracking, {self.start_stop_hotkey} to pause...')
            #self.writerow(['4', ts])
            self.write_event(TrackerEvent.TrackResume)
            self.is_tracking = True
        
        for eh in self.event_handlers:
            eh.on_pause_resume_event((not self.is_tracking), ts)
            
    def stop(self):
        self.mouse_listener.stop()

    def start(self):
        self.mouse_listener.start()
        #print(f'{self.start_stop_hotkey} to start.')

    def __init__(self) -> None:
        self.writers: List[TrackerDataWriter] = []
        self.event_handlers: List[TrackerEventHandler] = []

        self.is_tracking = False
        self.last_valid_mouse_down_ts: Optional[float] = None

        #self.start_stop_hotkey: str = 'ctrl+shift+a'
        #keyboard.add_hotkey(self.start_stop_hotkey, self.toggle) #TODO take this out of the class

        self.mouse_listener = mouse.Listener(on_click=self.on_click)

#TODO all outer layer stuff can access CLI, in CLI layer, addHandler for start, stop

# TODO move this to collect_cli
def main(args):
    data_filename: str = args[0] if args else 'raw_mouse_events.csv'
    print('Writing to ' + data_filename)

    with open(data_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        tracker = Tracker(writer)
        tracker.start()

        print("Esc to quit")
        keyboard.wait('esc')

        tracker.stop()

    print('Done! Wrote to ' + data_filename)

if __name__ == "__main__":
    main(sys.argv[1:])