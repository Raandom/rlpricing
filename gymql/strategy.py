import math

import numpy as np

from .store import UniformStore, PrioritizedStore

from gymql.ql import QLModel


class QLModel(object):
    def __init__(self, _, env, **kwargs):
        self.num_prices = math.floor(env.action_space.n)
        self.num_features = env.observation_space.shape[0]
        self.q = QLModel(self.num_features, self.num_prices, **kwargs)
        self.periodic_profits = []
        self.last_train = 0
        self.overall_eps = 0
        self.train_eps = kwargs.get("training_eps", 128)
        self.sample_size = kwargs.get("sample_size", 256)
        self.env = env
        store_p = kwargs.get("store_params", {})
        self.store = UniformStore(**store_p)\
            if kwargs.get("store_type", "uniform") == "uniform"\
            else PrioritizedStore(**store_p)

    def predict(self, obs):
        return [self.q.predict_values(self.normalize_state(np.asarray(obs)))], None

    def normalize_state(self, state):
        return (state / self.num_prices) - 0.5

    def learn(self, total_timesteps, log_interval, callback):
        state = self.normalize_state(self.env.reset())

        for i in range(0, total_timesteps):
            price = self.learn_single(state)
            last_state = state
            state, rewards, done, info = self.env.step([[price]])
            state = self.normalize_state(state)
            self.store.store(last_state, price, rewards, state)
            if done:
                break
            if i % log_interval == 0:
                callback({'step': i, 'losses': self.q.losses}, {})

    def learn_single(self, features):
        if self.last_train % self.train_eps == 0 and self.overall_eps >= self.store.size:
            self.last_train = 0
            samples, indices, weights = self.store.get_sample(self.sample_size)
            updates = self.q.train(samples, weights)
            self.store.update_weights(indices, updates)
        price = self.q.get_price(features)
        self.last_train += 1
        self.overall_eps += 1
        return price

    def get_periodic_profits(self):
        return self.periodic_profits
