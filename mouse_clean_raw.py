
import sys
import time
import csv
import pathlib

from typing import Optional, List


class PendingClick:

    def __init__(self, on_time: float, off_time: float):
        self.on_time = on_time
        self.off_time = off_time



def main(args):
    input_filename: str = 'raw_mouse_events.csv' #args[0]
    output_filename: str = pathlib.Path(input_filename).stem + '_cleaned.csv'

    on_streaks: List[int] = []
    off_streaks: List[int] = []

    with open(input_filename, 'r') as readf:
        with open(output_filename, 'w', newline='') as writef:
            r = csv.reader(readf)
            w = csv.writer(writef)

            curr_streak: int = 0
            last_event_type: str = ''
            last_event_time: float = 0
            pending_click: Optional[PendingClick] = None

            for readrow in r:
                event_type: str = readrow[0]
                event_time: float = float(readrow[1])

                if event_type == last_event_type:
                    curr_streak += 1
                elif curr_streak > 1:
                    if last_event_type == '0':
                        off_streaks.append(curr_streak)
                    elif last_event_type == '1':
                        on_streaks.append(curr_streak)
                    curr_streak = 0

                if event_type == '0':
                    if last_event_type == '1':
                        pending_click = PendingClick(on_time=last_event_time, off_time=event_time)
                if event_type == '1':
                    if pending_click:
                        pending_click.off_dur = event_time - last_event_time
                        w.writerow([pending_click.on_time,
                                    (pending_click.off_time - pending_click.on_time), 
                                    (event_time - pending_click.off_time)])
                elif event_type == '3': # paused, so pending clicks cancelled
                    pending_click = None
                elif event_type == '4': # resumed, may as well cancel pending clicks
                    pending_click = None

                last_event_type = event_type
                last_event_time = event_time

    print('Done! Wrote to ' + output_filename)
    print('Off streaks: ' + str(off_streaks))
    print('On streaks: ' + str(on_streaks))

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Need 1 args!")
        sys.exit(1)
    main(sys.argv[1:])