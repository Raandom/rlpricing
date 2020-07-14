from .getstrat import STRATEGIES
from .strategy import Strategy

class TwoBound(Strategy):
    def __init__(self, min_p, max_p, diff, hard_upper_limit=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min = min_p
        self.max = max_p
        self.diff = diff
        self.hard_upper_limit = hard_upper_limit

    def __call__(self, ref_p, conc_p):
        target = conc_p - self.diff
        if self.hard_upper_limit:
            target = min(target, self.max)
        return self.max if target < self.min else target


class CartelTwoBound(Strategy):
    def __init__(self, min_p, max_p, diff, cartel_p, hard_upper_limit=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.two_bound = TwoBound(min_p, max_p, diff, hard_upper_limit)
        self.min = min_p
        self.max = max_p
        self.cartel_p = cartel_p
        self.diff = diff

    def __call__(self, ref_p, conc_p):
        if round(conc_p) == round(self.cartel_p):
            return conc_p
        else:
            return self.two_bound(ref_p, conc_p)


class Fixed(Strategy):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p

    def __call__(self, ref_p, conc_p):
        return self.p


STRATEGIES['fixed'] = Fixed
STRATEGIES['cartel'] = CartelTwoBound
STRATEGIES['twobound'] = TwoBound
