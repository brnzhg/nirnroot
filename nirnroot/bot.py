from models.model import ModelGenProto

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from typing import List, Optional, Tuple, Protocol, Any

class BotClickGen:

    def gen_next(self) -> Tuple[float, float]:
        a = self.model_gen.gen_next()
        return (a[0], a[1])

    def reseed(self) -> None:
        self.model_gen.reseed()

    def __init__(self, model_gen: ModelGenProto):
        self.model_gen = model_gen

class ClickerProto(Protocol):
    
    def mouse_down(self):
        pass

    def mouse_up(self):
        pass

class Bot:
    click_gen: BotClickGen
    clicker: ClickerProto
    event_handler: float
    
    _running: bool

    def stop(self):
        self._running = False
        # TODO notify event_handler

    def reseed(self):
        self.click_gen.reseed()
    
    def click_loop(self):

        # TODO notify event_handler

        down_dur, up_dur = self.click_gen.gen_next()

        bad_gen_streak: int = 0
        while self._running:
            self.clicker.mouse_down()
            time.sleep(down_dur)
            self.clicker.mouse_up()
            click_done_ts:float = time.time()
            
            next_down_dur, next_up_dur = self.click_gen.gen_next()

            #TODO idk
            sleep_time: float = time.time() - click_done_ts

            if sleep_time > 1e-4:
                time.sleep(sleep_time)
            elif sleep_time < -5:
                raise Exception('big delay in generation')
            else:
                bad_gen_streak += 1

                if bad_gen_streak > 1:
                    raise Exception(f'{bad_gen_streak} bad gens consecutive')

        # notify eventhandler