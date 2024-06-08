import random
from enum import Enum, IntEnum
import itertools

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from typing import List, Optional, Tuple, Protocol, Any


#--------------
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, RootResults
#from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import RobustScaler, PowerTransformer, MinMaxScaler
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional
#----------


class ModelGenProto(Protocol):

    # single pair
    def gen_next(self) -> np.ndarray[Any]:
        ...

    # n pairs in 2d array
    def gen_next_n(self, n: int) -> np.ndarray[Any]:
        ...

    def reseed(self) -> None:
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

    

# Model to pickle
class SMDiscreteJointKde:

    transformer: PowerTransformer
    pass



class DiscreteCondOneDimSampler:
    
    def sample(self, indep_vars: NDArray[np.float]) -> float:
        #TODO have to convert indep_vars into buckets
        indep_idxs = tuple(self._lookup_1d_indep_bucket(i, v) 
                           for i, v in enumerate(indep_vars))

        dep_bin_weights = self.hist[indep_idxs]

        #TODO can optimize
        chosen_dep_bin_idx: int = \
            np.random.choice(self.choice_idxs, #np.arange(len(dep_bin_weights)), 
                             p=dep_bin_weights)
        
        return np.random.uniform(low=self.dep_bin_edges[chosen_dep_bin_idx], 
                                 high=self.dep_bin_edges[chosen_dep_bin_idx + 1])

    def _lookup_1d_indep_bucket(self, 
                                indep_col_idx: int, 
                                indep_var: np.float):
        bin_count = len(self.indep_bin_edges[indep_col_idx]) - 1 #self.indep_bins_counts[indep_col_idx]
        idx = np.digitize(indep_var, self.indep_bin_edges[indep_col_idx], right=True) - 1
        if idx == bin_count:
            return idx - 1
        return idx

    # hist - NDArray, (ind dim1 bins, ind dim2 bins, ..., dep bins)
    #   P(dep | dim1, dim2, ...)
    # dep_bin_edges - bin edges of dep var
    # indep_bin_edges - i-th ele has i-th indep dim bin edges 
    def __init__(self,
                 hist: NDArray[Any],
                 dep_bin_edges: NDArray[np.float],
                 indep_bin_edges: List[NDArray[np.float]]):
        self.hist: NDArray[Any] = hist
        self.dep_bin_edges: NDArray[np.float] = dep_bin_edges
        self.indep_bin_edges: List[NDArray[np.float]] = indep_bin_edges

        self.choice_idxs: NDArray[np.int] = np.arange(len(dep_bin_edges))

    
    
class DiscreteCondOneDimTrainer:


    # dep_data - 1d numpy array (obs)
    # indep_data - (obs, dims)
    def fit(self,
            dep_data: NDArray[np.float],
            indep_data: NDArray[Any]):
        
        num_indep_vars_supplied: int = indep_data.shape[1]
        if self.num_indep_vars != num_indep_vars_supplied:
            raise Exception(f"Expected {self.num_indep_vars} indep vars, data has {num_indep_vars_supplied}.")
        
        indep_bin_edges, indep_dig_data = \
            self._digitize_indep(indep_data)

        ckde = KDEMultivariateConditional(dep_data, 
                                               self.indep_dig_data,
                                               dep_type='c',
                                               indep_type=('o' * self.num_indep_vars),
                                               bw='normal_reference')
                                            
        dep_bin_edges, dep_bin_width = \
            np.linspace(self.sample_min, 
                        self.sample_max, 
                        num=self.dep_bins_count+1, 
                        retstep=True)


    def _make_hist(self):
        hist_l = list(self.indep_bins_counts)
        hist_l.append(self.dep_bins_count)
        hist_shape = tuple(hist_l)
        hist = np.zeros(hist_shape)

        for idxs in itertools.product(*(range(bncnt) for bncnt in self.indep_bins_counts)):
            #TODO could optimize probably, idk if reshape is needed
            curr_indep = np.array(list(idxs))
            cdf_at_edges = np.array([self.ckde.cdf(np.array(y).reshape(-1,1), 
                                          curr_indep.reshape(-1,1)) for y in self.dep_bin_edges])
            cdf_diffs = np.ediff1d(cdf_at_edges)
            cdf_diffs_sum = np.sum(cdf_diffs)
            
            for i in range(self.dep_bins_count):
                hist[idxs][i] = cdf_diffs[i] / cdf_diffs_sum
        return hist


    def _digitize_indep(self, indep_data: NDArray[Any]):
        
        indep_bin_edges: List[NDArray[np.float]] = []
        digs: List[NDArray[Any]] = []
        for i in range(self.num_indep_vars):
            bin_edges, dig = self._digitize_indep_col(indep_data, i)

            indep_bin_edges.append(bin_edges)
            digs.append(dig.reshape(-1,1))

        # append digs as column vectors to make digitized data
        indep_dig_data = np.concatenate(digs, axis=1)

        return indep_bin_edges, indep_dig_data

    # returns bin_edge array (for sampling to find buckets)
    #   and digitized column
    def _digitize_indep_col(self, indep_data: NDArray[Any], indep_col_idx: int):
        indep_col = indep_data[:, indep_col_idx]
        bin_count = self.indep_bins_counts[indep_col_idx]

        percentiles = np.linspace(0, 100, bin_count+1)
        # bin_count + 1 edges
        bin_edges = np.percentile(indep_col, percentiles)
        dig = np.digitize(indep_col, bin_edges, right=True) - 1
        dig[dig == bin_count] = bin_count - 1

        return bin_edges, dig

    def __init__(self,
                 dep_bin_count: int,
                 indep_bin_counts: List[int],
                 sample_min: float,
                 sample_max: float):
        
        self.dep_bin_count = dep_bin_count
        self.indep_bin_counts = indep_bin_counts
        self.sample_min = sample_min
        self.sample_max = sample_max

        self.num_indep_vars = len(indep_bin_counts)
    
# training data as raw array, do arranging of data, creating histogram, etc
class Trainer:
    pass


    