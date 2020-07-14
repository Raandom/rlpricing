import numpy as np

from .getstrat import STRATEGIES, get_strat
from .strategy import Strategy


class FullyRandomStrategy(Strategy):
    def __init__(self, min_p, max_p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min = min_p
        self.max = max_p

    def __call__(self, ref_p, conc_p):
        return np.floor(np.random.uniform(low=self.min, high=self.max))


class MixedRandomStrategy(Strategy):
    def __init__(self, strats, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.strats = [get_strat(el[0])(**el[1]) for el in strats]

    def __call__(self, ref_p, conc_p):
        return np.random.choice(self.strats, p=self.p)(ref_p, conc_p)


class NoisedStrategy(Strategy):
    def __init__(self, strat, strat_params, stddev, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stddev = stddev
        self.strat = get_strat(strat)(**strat_params)

    def __call__(self, *args, **kwargs):
        return np.round(np.random.normal(self.strat(*args, **kwargs), self.stddev), decimals=1)


STRATEGIES['random'] = FullyRandomStrategy
STRATEGIES['mixedrandom'] = MixedRandomStrategy
STRATEGIES['noised'] = NoisedStrategy
