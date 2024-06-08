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
            indep_data: NDArray[Any]) -> DiscreteCondOneDimSampler:
        
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
            np.linspace(self.sample_bounds[0], 
                        self.sample_bounds[1], 
                        num=self.dep_bin_count+1, 
                        retstep=True)

        hist = self._make_hist_from_ckde(ckde, dep_bin_edges)

        return DiscreteCondOneDimSampler(hist, dep_bin_edges, indep_bin_edges)


    def _make_hist_from_ckde(self, ckde: KDEMultivariateConditional, dep_bin_edges: NDArray[np.float]) -> NDArray[Any]:
        hist_l = list(self.indep_bin_counts)
        hist_l.append(self.dep_bin_count)
        hist_shape = tuple(hist_l)
        hist = np.zeros(hist_shape)

        for idxs in itertools.product(*(range(bncnt) for bncnt in self.indep_bin_counts)):
            #TODO could optimize probably, idk if reshape is needed
            curr_indep = np.array(list(idxs))
            cdf_at_edges = np.array([ckde.cdf(np.array(y).reshape(-1,1), 
                                              curr_indep.reshape(-1,1)) for y in dep_bin_edges])
            cdf_diffs = np.ediff1d(cdf_at_edges)
            cdf_diffs_sum = np.sum(cdf_diffs)
            
            for i in range(self.dep_bin_count):
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
                 sample_bounds: Tuple[float, float]):
        
        self.dep_bin_count = dep_bin_count
        self.indep_bin_counts = indep_bin_counts
        self.sample_bounds = sample_bounds

        self.num_indep_vars = len(indep_bin_counts)

    
# Model to pickle
class SMDiscreteJointKde:

    transformer: PowerTransformer
    down_sampler: DiscreteCondOneDimSampler
    up_sampler: DiscreteCondOneDimSampler

    

    pass


# training data as raw array, do arranging of data, creating histogram, etc
class Trainer:
    pass

