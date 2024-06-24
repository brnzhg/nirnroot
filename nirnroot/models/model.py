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
    def gen_next(self) -> NDArray[Any]:
        ...

    # n pairs in 2d array
    def gen_next_n(self, n: int) -> NDArray[Any]:
        ...

    def reseed(self) -> None:
        ...


class ModelGenFactoryProto(Protocol):
    
    def make_gen(self) -> ModelGenProto:
        ...


class ModelTrainerProto(Protocol):
    
    def fit(self, data: NDArray[Any]) -> ModelGenFactoryProto:
        ...


    