
import sys
import time
import datetime
import csv

from pathlib import Path

import random
from enum import Enum, IntEnum
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from typing import List, Optional, Tuple, Protocol, Any

#from _typeshed import SupportsWrite
#import shutil

#--------------
import numpy as np
from scipy.optimize import brentq, RootResults
#from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import RobustScaler, PowerTransformer, MinMaxScaler
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional
#----------



@dataclass_json
@dataclass
class ModelMetadata:
    id: str
    name: str
    model_type: str
    training_id: str
    

class ModelGenProto(Protocol):

    # single pair
    def gen_next(self) -> np.ndarray[Any]:
        ...

    # n pairs in 2d array
    def gen_next_n(self, n: int) -> np.ndarray[Any]:
        ...

    def reseed(self) -> None:
        ...


class BotClickGen:

    def gen_next(self) -> Tuple[float, float]:
        pass

    def reseed(self) -> None:
        pass

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

        down_dur, up_dur = self.click_gen

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


# implementation sfor each model specifically
# this is CLI crap layer, each model logic dont need to know this, dep other way
# bridges general CLI layer and modelgen layer, it's the most specific
class ModelGenLoaderProto(Protocol):
    model_type: str # tells them what type it can load

    #TODO could take in ModelMetadata or just load itself 
    def load_model_gen(self, md: ModelMetadata, model_path: Path) -> ModelGenProto:
        ...
#------------------------
 
# model to pickle 
class SMSlowCondKde:
    transformer: PowerTransformer
    down_ckde: KDEMultivariateConditional 
    up_cdke: KDEMultivariateConditional
    num_lags: int

    down_icdf_low_bound: float
    down_icdf_high_bound: float
    up_icdf_low_bound: float
    up_icdf_high_bound: float
    icdf_tol: float


    def cdf_conditional(self, x, y_target):
    # cdf of kde conditional on x, evaluated at y_target
        return self.ckde.cdf(np.array(y_target).reshape((-1,1)), np.array(x).reshape((-1,1)))

    # inverse-transform-sampling
    def _sample_conditional_helper(self, ckde: KDEMultivariateConditional, x, icdf_low_bound: float, icdf_high_bound: float):
    # sample conditional on x
        u = np.random.random()
        # 1-d root-finding
        def func(y):
            return self.cdf_conditional(x, y) - u

        #TODO handle failure and max iter
        sample_y = brentq(func, icdf_low_bound, icdf_high_bound, xtol=self.icdf_tol)  

        #try:
        #    sample_y = brentq(func, -999, 999, maxiter=5)  
        #except:
            #pass
        
        # constants need to be sign-changing for the function
        return sample_y

    def sample_down_conditional_transformed(self, lags):
        return self._sample_conditional_helper(self.down_ckde, lags, self.down_icdf_low_bound, self.down_icdf_high_bound)

    def sample_up_conditional_transformed(self, down_and_lags):
        return self._sample_conditional_helper(self.up_ckde, down_and_lags, self.up_icdf_low_bound, self.up_icdf_high_bound)

    def inverse_transform(self, trfmed):
        return self.transformer.inverse_transform(trfmed)
        
    
# model gen
class SMSlowCondKdeGen(ModelGenProto):

    model: SMSlowCondKde
    num_lags: int
   
    def gen_next_transformed(self) -> Tuple[float, float]:


       down_sample = self.model.sample_down_conditional_transformed(self._last_lags)

       down_and_lags = np.concatenate([np.array([down_sample]), self._last_lags])
       
       up_sample = self.model.sample_up_conditional_transformed(down_and_lags)

       # drop last of the downup lag pairs
       self._last_lags = np.concatenate([np.array([down_sample, up_sample]), self._last_lags[:-2]])

       return (down_sample, up_sample)

    # gen down float and up float (shitty)
    # return numpy array
    def gen_next(self) -> np.ndarray[Any]:
        return self.gen_next_n(1)[0]

    # return numpy array
    def gen_next_n(self, n: int) -> np.ndarray[Any]:
        l = []
        for _ in range(n):
            l.append(self.gen_next_transformed())
        return self.model.inverse_transform(np.array(l))


    #TODO could do this if sampling fails
    def reseed(self):
        new_seed_idx: int = random.randrange(self.model.num_lags, self.train_data.shape[0])
        self._last_lags = self.get_last_lags_from_idx(new_seed_idx)


    def get_last_lags_from_idx(self, idx: int):
        return np.concatenate([self.train_data[idx - i,:] for i in range(self.num_lags)])
        

    def __init__(self, model: SMSlowCondKde, train_data, seed_idx: Optional[int]):
        self.model = model
        self.num_lags = model.num_lags
        self.train_data = train_data

        if seed_idx:
            self._seed_idx = seed_idx
        else:
            self._seed_idx = random.randrange(self.model.num_lags, train_data.shape[0])
        self._last_lags = self.get_last_lags_from_idx(self._seed_idx)

    
