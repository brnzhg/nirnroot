import sys
import time
import csv
from pynput import mouse
import keyboard

from typing import Optional

class Tracker:

    def on_click(self, x, y, button, pressed):
        if not self.is_tracking:
            return
        self.writer.writerow(['1' if pressed else '0', time.time()])

    def toggle(self):
        if self.is_tracking:
            print(f'Stopping tracking, {self.start_stop_hotkey} to resume...')
            self.writer.writerow(['3', time.time()])
            self.is_tracking = False
        else:
            print(f'Starting tracking, {self.start_stop_hotkey} to pause...')
            self.writer.writerow(['4', time.time()])
            self.is_tracking = True

    def stop(self):
        self.mouse_listener.stop()

    def start(self):
        self.mouse_listener.start()
        print(f'{self.start_stop_hotkey} to start.')

    def __init__(self, writer):
        self.writer = writer

        self.start_stop_hotkey: str = 'ctrl+shift+a'
        self.is_tracking: bool = False
        self.last_valid_mouse_down_ts: Optional[float] = None

        keyboard.add_hotkey(self.start_stop_hotkey, self.toggle) #TODO take this out of the class

        self.mouse_listener = mouse.Listener(on_click=self.on_click)

#TODO all outer layer stuff can access CLI, in CLI layer, addHandler for start, stop

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