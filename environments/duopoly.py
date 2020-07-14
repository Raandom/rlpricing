import gym
import numpy as np
from gym import spaces

from simtools import update_ref, CustomerSimulator, get_history, store_history
from strategies import get_strat


class SpaceHandler(object):
    def __init__(self, min_p, max_p, step_size):
        self.min_p = min_p
        self.max_p = max_p
        self.step_size = step_size

    def fix_action(self, price, comp_price):
        raise NotImplementedError()

    def get_action_space(self):
        raise NotImplementedError()


class ContinuousSpaceHandler(SpaceHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.space_length = (self.max_p - self.min_p) / 2

    def get_action_space(self):
        return spaces.Box(low=-self.space_length, high=self.space_length, shape=(1,))

    def fix_action(self, price, comp_price):
        return round(self.min_p + self.space_length + price, 2)


class DiscreteRelativeSpaceHandler(SpaceHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.space_length = ((self.max_p - self.min_p) / self.step_size) + 1

    def get_action_space(self):
        return spaces.Discrete(self.space_length)

    def fix_action(self, price, comp_price):
        return self.min_p + (price * self.step_size) if (price < self.space_length - 1) else comp_price - 1


class DiscreteSpaceHandler(SpaceHandler):
    def get_action_space(self):
        space_length = (self.max_p - self.min_p) / self.step_size
        return spaces.Discrete(space_length)

    def fix_action(self, price, comp_price):
        return self.min_p + (price * self.step_size)


def get_action_space_handler(mode):
    if mode == "continuous":
        return ContinuousSpaceHandler
    elif mode == "discrete_relative": # Note that this requires use_relative to be set in the model
        return DiscreteRelativeSpaceHandler
    else:
        return DiscreteSpaceHandler


class MarketEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, min_p, max_p, competitor, comp_params, sim_params, h, w, mode="continuous", *args, **kwargs):
        super(MarketEnv, self).__init__()

        self.comp = get_strat(competitor)(**comp_params)
        self.mode = mode
        self.min_p = min_p
        self.max_p = max_p
        self.step_size = kwargs.pop("step_size", 0.1)

        self.space_handler = get_action_space_handler(mode)(min_p, max_p, self.step_size)

        self.action_space = self.space_handler.get_action_space()

        self.observation_space = spaces.Box(low=0, high=100,
                                            shape=(2, ), dtype=np.float)

        # Calculation params (Demand sim, length of an episode, factor for running average ref price)
        self.w = w
        self.h = h
        self.sim = CustomerSimulator(**sim_params)

        # State dependent variables (we should reset those when resetting the env)
        self.price_b = 30
        self.ref_p = 30
        self.overall_profits_a = 0
        self.overall_profits_b = 0

        self.hist_a = get_history()
        self.hist_b = get_history()

        self.episodes = 0

    def fix_action(self, price, comp_price):
        return self.space_handler.fix_action(price, comp_price)

    def step(self, action):
        # First: a changes prices:
        price_a = self.fix_action(action[0], self.price_b)
        # First thing to do: Reshape action as Model only supports symmetric space
        self.hist_a.prices.append(price_a)
        self.ref_p = update_ref(self.ref_p, price_a, self.w)

        sales_a = self.sim(self.h, price_a, self.price_b, self.ref_p)
        self.hist_a.sales_first.append(sales_a)
        profits_a = sales_a * price_a
        sales_b = self.sim(self.h, self.price_b, price_a, self.ref_p)
        self.hist_b.sales_first.append(sales_b)
        profits_b = sales_b * self.price_b

        # Second: b changes prices:
        self.price_b = self.comp(self.ref_p, price_a)
        self.hist_b.prices.append(self.price_b)
        self.ref_p = update_ref(self.ref_p, self.price_b, self.w)

        sales_a = self.sim(1 - self.h, price_a, self.price_b, self.ref_p)
        self.hist_a.sales_second.append(sales_a)
        profits_a += sales_a * price_a
        sales_b = self.sim(1 - self.h, self.price_b, price_a, self.ref_p)
        self.hist_b.sales_second.append(sales_b)
        profits_b += sales_b * self.price_b

        self.overall_profits_a += profits_a
        self.overall_profits_b += profits_b
        self.hist_a.profits.append(profits_a)
        self.hist_b.profits.append(profits_b)

        self.episodes += 1

        return np.asarray([self.ref_p, self.price_b]), profits_a, False, {}

    def reset(self):
        self.price_b = 30
        self.ref_p = 30

        self.hist_a = get_history()
        self.hist_b = get_history()

        self.overall_profits_b = 0
        self.overall_profits_a = 0

        self.episodes = 0

        return np.asarray([self.ref_p, self.price_b])

    def render(self, mode='human', close=False):
        pass # print("Current standing {}/{}".format(self.overall_profits_a, self.overall_profits_b))

    def store_history(self, file, *args, **kwargs):
        store_history(
            file,
            self.overall_profits_a, self.overall_profits_b,
            self.hist_a, self.hist_b, *args, **kwargs
        )

    def display(self, text):
        print(text.format(self.overall_profits_a, self.overall_profits_b))


class DoubleMarketEnv(object):
    def __init__(self, min_p, max_p, h, w, sim_params, mode="continuous", mode_b="continuous", *args, **kwargs):

        self.min_p = min_p
        self.max_p = max_p
        self.step_size = kwargs.pop("step_size", 0.1)
        self.space_handler_a = get_action_space_handler(mode)(min_p, max_p, self.step_size)
        self.space_handler_b = get_action_space_handler(mode_b)(min_p, max_p, self.step_size)

        self.observation_space = spaces.Box(low=0, high=100,
                                        shape=(2, ), dtype=np.float)

        # Calculation params (Demand sim, length of an episode, factor for running average ref price)
        self.w = w
        self.h = h
        self.sim = CustomerSimulator(**sim_params)

        self.a = None
        self.b = None

        # State dependent variables (we should reset those when resetting the env)
        self.episodes = 0
        self.price_a = 30
        self.price_b = 30
        self.ref_p = 30

        # Training vars (reset those when one epoch of training is done)
        self.overall_profits_a = 0
        self.overall_profits_b = 0
        self.hist_a = get_history()
        self.hist_b = get_history()

    def get_env(self, a=True):
        wrapping_env = self
        space_handler = self.space_handler_a if a else self.space_handler_b

        class Env(gym.Env):
            def __init__(self):
                self.action_space = space_handler.get_action_space()
                self.observation_space = wrapping_env.observation_space

            def step(self, action):
                return wrapping_env.step(a, action)

            def reset(self):
                return wrapping_env.reset(a)

        return Env()

    def register_comp(self, comp, a=True):
        if a:
            self.a = comp
        else:
            self.b = comp

    def fix_action(self, price, a, comp_price):
        if a:
            return self.space_handler_a.fix_action(price, comp_price)
        else:
            return self.space_handler_b.fix_action(price, comp_price)

    def finish_training(self):
        self.overall_profits_a = 0
        self.overall_profits_b = 0
        self.hist_a = get_history()
        self.hist_b = get_history()

    def half_step(self, a, action, first):
        action = self.fix_action(action, a, self.price_b if a else self.price_a)

        self.ref_p = update_ref(self.ref_p, action, self.w)

        if a:
            self.price_a = action
        else:
            self.price_b = action
        (self.hist_a.prices if a else self.hist_b.prices).append(action)

        h = self.h if first else 1 - self.h
        sales_a = self.sim(h, self.price_a, self.price_b, self.ref_p)
        (self.hist_a.sales_first if first else self.hist_a.sales_second).append(sales_a)
        profits_a = sales_a * self.price_a

        sales_b = self.sim(h, self.price_b, self.price_a, self.ref_p)
        (self.hist_b.sales_first if first else self.hist_b.sales_second).append(sales_a)
        profits_b = sales_b * self.price_b

        return profits_a, profits_b

    def step(self, a, action):
        # First: a changes prices:
        state = np.asarray([[self.ref_p, self.price_b]])
        price_a = action[0] if a else self.a.predict(state)[0][0][0]

        profits_a, profits_b = self.half_step(True, price_a, True)

        # Second: b changes prices:
        state = np.asarray([[self.ref_p, self.price_a]])
        price_b = action[0] if not a else self.b.predict(state)[0][0][0]

        profits_a_second, profits_b_second = self.half_step(False, price_b, False)
        profits_a += profits_a_second
        profits_b += profits_b_second

        self.hist_a.profits.append(profits_a)
        self.hist_b.profits.append(profits_b)

        self.episodes += 1

        return np.asarray([self.ref_p, self.price_b if a else self.price_a]), profits_a if a else profits_b, False, {}

    def reset(self, a):
        self.price_a = 30
        self.price_b = 30
        self.ref_p = 30

        self.episodes = 0

        return np.asarray([self.ref_p, self.price_b if a else self.price_a])

    def store_history(self, file, *args, **kwargs):
        store_history(
            file,
            self.overall_profits_a, self.overall_profits_b,
            self.hist_a, self.hist_b, *args, **kwargs
        )

    def display(self, text):
        print(text.format(self.overall_profits_a, self.overall_profits_b))
