import gym
import numpy as np
from gym import spaces

from simtools import get_history, store_oligopoly_history
from strategies import get_strat


class OligopolyCustomerSimulator(object):
    def __init__(self, num_competitors, customer_arrival_time=0.1, price_factors=None, biases=None, cus_stddev=2, *args, **kwargs):
        self.num_comp = num_competitors
        self.customer_arrival_time = customer_arrival_time
        self.price_factors = np.asarray(price_factors) if price_factors else np.ones((self.num_comp + 1, ))
        self.biases = np.asarray(biases) if biases else np.zeros((self.num_comp + 1, ))
        self.cus_stddev = cus_stddev
        self.time = 0

    def __call__(self, step, prices):
        time = self.time
        dist = (self.customer_arrival_time * 0.1)
        low = self.customer_arrival_time - dist
        high = self.customer_arrival_time + dist
        sales = np.zeros_like(prices)
        while True:
            next_cus = np.random.uniform(low, high)
            time += next_cus
            if time > (self.time + step):
                break
            chosen = self.rank(prices)
            sales[chosen] += 1

        self.time = time - next_cus

        return sales

    def rank(self, prices):
        return np.argmin(np.random.normal((prices * self.price_factors) + self.biases, self.cus_stddev))


class OligopolyEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, min_p, max_p, comp_params, sim_params, mode="continuous", agent_intervall=1.0, time_stdev=0.05, *args, **kwargs):
        super(OligopolyEnv, self).__init__()

        self.agent_intervall = agent_intervall
        self.time_stdev = time_stdev

        self.mode = mode
        self.min_p = min_p
        self.max_p = max_p
        self.step_size = kwargs.pop("step_size", 0.1)
        if mode == "continuous":
            self.space_length = (self.max_p - self.min_p) / 2
            self.action_space = spaces.Box(low=-self.space_length, high=self.space_length, shape=(1,))
        else:
            self.space_length = (self.max_p - self.min_p) / self.step_size
            self.action_space = spaces.Discrete(self.space_length)

        self.observation_space = spaces.Box(low=0, high=100,
                                            shape=(len(comp_params), ), dtype=np.float)

        # Calculation params (Demand sim, length of an episode, factor for running average ref price)
        self.sim = OligopolyCustomerSimulator(len(comp_params), **sim_params)

        # State dependent variables (we should reset those when resetting the env
        self.episodes = 0
        self.time = 0
        self.comp = [get_strat(comp.get("type"))(**comp) for comp in comp_params]
        self.timings = np.asarray([self.agent_intervall] + [c.intervall for c in self.comp])
        self.timings = np.random.normal(self.timings, self.time_stdev)
        self.prices = np.ones_like(self.timings)
        self.history = [get_history() for _ in self.prices]
        self.overall_profits = np.zeros_like(self.prices)

    def fix_action(self, price, comp_price):
        if self.mode == "continuous":
            return round(self.min_p + self.space_length + price, 2)
        else:
            return self.min_p + (price * self.step_size)

    def compute_rewards(self, step):
        sales = self.sim(step, self.prices)
        profit = sales * self.prices
        return profit, sales

    def step(self, action):
        # Update a's price and compute rewards for the last ep
        price_a = self.fix_action(action[0], min(self.prices))
        self.history[0].prices.append([self.time, price_a])
        self.prices[0] = price_a
        self.timings[0] = np.random.normal(self.agent_intervall, self.time_stdev)

        profits = np.zeros_like(self.timings)
        sales = np.zeros_like(self.timings)

        # Then: We iterate through our competitors
        while True:
            # Set everything below zero to zero to ensure nobody would have updated
            # in the past
            self.timings[self.timings < 0] = 0
            opp_idx = np.argmin(self.timings)
            step = self.timings[opp_idx]
            self.time += step

            # Compute earnings in that period
            profits_ep, sales_ep = self.compute_rewards(step)
            profits += profits_ep
            sales += sales_ep

            self.timings = self.timings - step
            self.timings[opp_idx] = np.random.normal(self.comp[opp_idx - 1].intervall, self.time_stdev)
            if opp_idx == 0:
                break

            # If its not up to the agent to update, the competitor updates
            self.prices[opp_idx] = self.comp[opp_idx - 1].get_for_oligopoly(self.ref_p, np.delete(self.prices, opp_idx))
            self.history[opp_idx].prices.append([self.time, self.prices[opp_idx]])

        self.episodes += 1

        for i in range(0, len(profits)):
            self.history[i].sales.append(sales[i])
            self.history[i].profits.append(profits[i])
        self.overall_profits += profits

        return self.prices[1:], profits[0], False, {}

    def reset(self):
        self.episodes = 0
        self.timings = np.asarray([self.agent_intervall] + [c.intervall for c in self.comp])
        self.prices = np.ones_like(self.prices)
        self.history = [get_history() for _ in self.prices]
        self.overall_profits = np.zeros_like(self.prices)
        self.ref_p = 30
        self.time = 0

        return self.prices[1:]

    def render(self, mode='human', close=False):
        pass

    def store_history(self, file, *args, **kwargs):
        store_oligopoly_history(
            file, self.history
        )

    def display(self, text):
        print(text.format(self.overall_profits[0], np.median(self.overall_profits[1:])))
