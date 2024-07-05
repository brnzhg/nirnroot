import time
import datetime

from typing import List, Optional, Tuple, Protocol

from models.domain import TrackerEvent


class TrackerDataWriterProto(Protocol):
    def write_event(self, e: TrackerEvent, timestamp: float) -> None:
        ...

# TODO this is for printing to screen
class TrackerEventHandlerProto(Protocol):
    def on_pause_resume_event(self, is_pause: bool, ts: float) -> None:
        ...

        
class Tracker:

    is_tracking: bool
    
    def add_event_handler(self, handler: TrackerEventHandlerProto):
        self.event_handlers.append(handler)

    def add_writer(self, writer: TrackerDataWriterProto):
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


        self.mouse_listener = mouse.Listener(on_click=self.on_click)
    