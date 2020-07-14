import numpy as np


class Strategy(object):
    def __init__(self, *args, **kwargs):
        self.intervall = kwargs.get("intervall", None)
        if not self.intervall:
            self.intervall = np.random.uniform(0.3, 1.7)

    def __call__(self, ref_p, opponent):
        raise NotImplementedError()

    def get_for_oligopoly(self, ref_p, prices):
        # This is a default that might be adjusted on a strategy
        # per strategy basis, but using the duopoly strategy
        # with the lowest price in the market seems reasonable right now
        return self.__call__(ref_p, np.min(prices))
