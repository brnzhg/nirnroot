
import random

from typing import List, Optional, Tuple, Protocol, Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, RootResults
from sklearn.preprocessing import RobustScaler, PowerTransformer, MinMaxScaler
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional

from model import ModelGenProto


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

    

