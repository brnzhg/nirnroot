import random
from enum import Enum, IntEnum
import itertools
from collections import deque

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from typing import List, Optional, Tuple, Protocol, Any


#--------------
import numpy as np
from numpy import float64, int64
import numpy.typing as npt
from numpy.typing import NDArray #, float64, int64

from sklearn.preprocessing import RobustScaler, PowerTransformer, MinMaxScaler
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional

from .model import ModelGenProto, ModelGenFactoryProto, ModelTrainerProto


class DiscreteCondOneDimSampler:
    
    def sample(self, indep_vars: NDArray[float64]) -> float:
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
                                indep_var: float64):
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
                 dep_bin_edges: NDArray[float64],
                 indep_bin_edges: List[NDArray[float64]]):
        self.hist: NDArray[Any] = hist
        self.dep_bin_edges: NDArray[float64] = dep_bin_edges
        self.indep_bin_edges: List[NDArray[float64]] = indep_bin_edges

        self.choice_idxs: NDArray[int64] = np.arange(len(dep_bin_edges) - 1)

    
    
class DiscreteCondOneDimTrainer:


    # dep_data - 1d numpy array (obs)
    # indep_data - (obs, dims)
    def fit(self,
            dep_data: NDArray[float64],
            indep_data: NDArray[Any],
            sample_bounds: Tuple[float, float]) -> DiscreteCondOneDimSampler:
        
        num_indep_vars_supplied: int = indep_data.shape[1]
        if self.num_indep_vars != num_indep_vars_supplied:
            raise Exception(f"Expected {self.num_indep_vars} indep vars, data has {num_indep_vars_supplied}.")
        
        indep_bin_edges, indep_dig_data = \
            self._digitize_indep(indep_data)

        ckde = KDEMultivariateConditional(dep_data,
                                          indep_dig_data,
                                          dep_type='c',
                                          indep_type=(
                                              'o' * self.num_indep_vars),
                                          bw='normal_reference')
                                            
        dep_bin_edges, dep_bin_width = \
            np.linspace(sample_bounds[0], 
                        sample_bounds[1], 
                        num=self.dep_bin_count+1, 
                        retstep=True)

        hist = self._make_hist_from_ckde(ckde, dep_bin_edges)

        return DiscreteCondOneDimSampler(hist, dep_bin_edges, indep_bin_edges)


    def _make_hist_from_ckde(self, 
                             ckde: KDEMultivariateConditional, 
                             dep_bin_edges: NDArray[float64]) -> NDArray[Any]:
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
        
        indep_bin_edges: List[NDArray[float64]] = []
        digs: List[NDArray[Any]] = []
        for i in range(self.num_indep_vars):
            bin_edges, dig = self._digitize_indep_col(indep_data, i)

            indep_bin_edges.append(bin_edges)
            digs.append(dig)

        # append digs as column vectors to make digitized data
        indep_dig_data = np.column_stack(digs)

        return indep_bin_edges, indep_dig_data

    # returns bin_edge array (for sampling to find buckets)
    #   and digitized column
    def _digitize_indep_col(self, indep_data: NDArray[Any], indep_col_idx: int):
        indep_col = indep_data[:, indep_col_idx]
        bin_count = self.indep_bin_counts[indep_col_idx]

        percentiles = np.linspace(0, 100, bin_count+1)
        # bin_count + 1 edges
        bin_edges = np.percentile(indep_col, percentiles)
        dig = np.digitize(indep_col, bin_edges, right=True) - 1
        dig[dig == bin_count] = bin_count - 1

        return bin_edges, dig

    def __init__(self,
                 dep_bin_count: int,
                 indep_bin_counts: Tuple[int, ...]):
        
        self.dep_bin_count = dep_bin_count
        self.indep_bin_counts = indep_bin_counts

        self.num_indep_vars = len(indep_bin_counts)

    
# Model to pickle, also pickle the Trainer in case you want to retrain
class SMDiscreteJointKde:

    transformer: PowerTransformer
    down_sampler: DiscreteCondOneDimSampler
    up_sampler: DiscreteCondOneDimSampler
    num_down_lags: int
    num_up_lags: int


    def sample_down_transformed(self, lags: NDArray[float64]):
        return self.down_sampler.sample(lags)

    def sample_up_conditional_transformed(self, lags: NDArray[float64]):
        return self.up_sampler.sample(lags)

    # trfmed shape (obs, 2), first col is down, second col up
    def inverse_transform(self, trfmed: NDArray[Any]):
        return self.transformer.inverse_transform(trfmed)
    

    def __init__(self, 
                 transformer: PowerTransformer, 
                 down_sampler: DiscreteCondOneDimSampler, 
                 up_sampler: DiscreteCondOneDimSampler,
                 num_down_lags: int,
                 num_up_lags: int):
        self.transformer = transformer
        self.down_sampler = down_sampler
        self.up_sampler = up_sampler

        self.num_down_lags = num_down_lags
        self.num_up_lags = num_up_lags


# training data as raw array, do arranging of data, creating histogram, etc
class Trainer:

    down_trainer: DiscreteCondOneDimTrainer
    up_trainer: DiscreteCondOneDimTrainer

    rel_sample_min: float
    rel_sample_max: float

    num_down_lags: int
    num_up_lags: int

    
    def fit(self, data: NDArray[Any]) -> SMDiscreteJointKde:

        min_obs_pre_transform = \
            self.rel_sample_min * np.array([np.min(data[:,0]), np.min(data[:,1])])
        max_obs_pre_transform = \
            self.rel_sample_max * np.array([np.max(data[:,0]), np.max(data[:,1])])
        
        transformer: PowerTransformer = PowerTransformer(method='box-cox')
        tdata = transformer.fit_transform(data)
        sample_min = transformer.transform([min_obs_pre_transform])[0]
        sample_max = transformer.transform([max_obs_pre_transform])[0]

        down_dep_data, down_indep_data = self._get_train_arrs(tdata, True)
        down_sampler = self.down_trainer.fit(down_dep_data, down_indep_data, (sample_min[0], sample_max[0]))

        up_dep_data, up_indep_data = self._get_train_arrs(tdata, False)
        up_sampler = self.up_trainer.fit(up_dep_data, up_indep_data, (sample_min[1], sample_max[1]))

        return SMDiscreteJointKde(transformer, down_sampler, up_sampler, self.num_down_lags, self.num_up_lags)


    def _get_train_arrs(self, tdata: NDArray[Any], is_down: bool) -> Tuple[NDArray[float64], NDArray[Any]]:
        num_obs: int = tdata.shape[0]
        # subtract the last down from the up lags
        num_prev_click_lags: int = self.num_down_lags if is_down else self.num_up_lags - 1

        num_prev_clicks: int = \
            0 if num_prev_click_lags <= 0 else (num_prev_click_lags + 1) // 2

        num_obs_lagged: int = num_obs - num_prev_clicks
            
        # most lagged to least lagged
        cols_l = []
        start_col = num_prev_click_lags % 2
        
        for i in range(num_prev_click_lags):
            i2 = (i + start_col) // 2
            j = (i + start_col) % 2
            cols_l.append(tdata[i2:(i2 + num_obs_lagged), j])
        #for i, j in zip(range(num_prev_click_lags), itertools.cycle([0, 1])):
        #    i2 = i // 2
            #slice_start = num_prev_click_lags - i2
        #    cols_l.append(tdata[i2:(i2 + num_obs_lagged), j])

        if not is_down:
            # append non-lagged down data
            cols_l.append(tdata[num_prev_clicks:,0])

        indep_data = np.column_stack(cols_l)
        dep_data = tdata[num_prev_clicks:, (0 if is_down else 1)]

        return dep_data, indep_data
    

    def __init__(self,
                 down_trainer: DiscreteCondOneDimTrainer,
                 up_trainer: DiscreteCondOneDimTrainer,
                 rel_sample_min: float,
                 rel_sample_max: float):
        self.down_trainer = down_trainer
        self.up_trainer = up_trainer

        self.rel_sample_min = rel_sample_min
        self.rel_sample_max = rel_sample_max
        
        self.num_down_lags = len(down_trainer.indep_bin_counts)
        self.num_up_lags = len(up_trainer.indep_bin_counts)




class SMDiscreteJointKdeGen(ModelGenProto):

    model: SMDiscreteJointKde
    train_data: NDArray[Any] # (obs, [down, up])
    num_lag_pairs: int
    lag_queue: List[float]
    seed_idx: int

    def gen_next_transformed(self) -> Tuple[float, float]:
        # TODO
        down_indep = np.array(self.lag_queue[-self.model.num_down_lags:])
        down_sample = self.model.down_sampler.sample(down_indep)

        self.lag_queue.append(down_sample)
        #self.lag_queue.pop(0)

        up_indep = np.array(self.lag_queue[-self.model.num_up_lags:])
        up_sample = self.model.up_sampler.sample(up_indep)

        self.lag_queue.append(up_sample)

        self.lag_queue = self.lag_queue[2:]

        return (down_sample, up_sample)

    # gen down float and up float (shitty)
    # return numpy array
    def gen_next(self) -> NDArray[float64]:
        return self.gen_next_n(1)[0]

    # shape (num_clicks, [down, up])
    def gen_next_n(self, n: int) -> NDArray[Any]:
        l = list(self.gen_next_transformed() for _ in range(n))
        return self.model.inverse_transform(np.array(l))


    def reseed(self):
        new_seed_idx: int = random.randrange(self.model.num_lag_pairs, self.train_data.shape[0])
        self.set_seed_idx(new_seed_idx)

    def set_seed_idx(self, idx: int):
        self.seed_idx = idx
        
        self.lag_queue.clear()
        for r in self.train_data[(idx - self.num_lag_pairs + 1):(idx + 1), :]:
            rt = self.model.transformer.transform(r.reshape(1, -1))
            self.lag_queue.append(rt[0, 0]) # down first
            self.lag_queue.append(rt[0, 1]) # then up


    # TODO in training and vars, goes lag2, lag1, current, so latest at front of list
    def __init__(self, 
                 model: SMDiscreteJointKde, 
                 train_data: NDArray[Any], 
                 seed_idx: Optional[int]):
        self.model = model
        self.train_data = train_data

        # one of the up lags comes from newly generated down
        num_lags = max(model.num_up_lags - 1, model.num_down_lags)
        if num_lags % 2 > 0:
            num_lags += 1
        self.num_lag_pairs = num_lags // 2

        self.lag_queue = []
        self.set_seed_idx(seed_idx or (train_data.shape[0] - 1))
        
        
class ModelGenFactory(ModelGenFactoryProto):
    model: SMDiscreteJointKde
    train_data: NDArray[Any]
    
    def make_gen(self) -> SMDiscreteJointKdeGen:
        return SMDiscreteJointKdeGen(self.model, self.train_data, None)

    def __init__(self,
                 model: SMDiscreteJointKde,
                 train_data: NDArray[Any]) -> None:
        self.model = model
        self.train_data = train_data
    

class ModelTrainer(ModelTrainerProto):
    trainer: Trainer

    def fit(self, data: NDArray[Any]) -> ModelGenFactory:
        model = self.trainer.fit(data)
        return ModelGenFactory(model, data)

    def __init__(self, trainer: Trainer) -> None:
        self.trainer = trainer
    
    