import collections
import csv
import datetime
import json
import os
import sys
import psutil

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from stable_baselines import SAC, DDPG
from stable_baselines.ddpg.policies import MlpPolicy as DDPGMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.sac import MlpPolicy

from environments.oligopoly import OligopolyEnv
from randommodel import RandomModel
from simtools import update
from environments.duopoly import MarketEnv
from gymql.strategy import QLModel

def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

if len(sys.argv) > 1:
    files = sys.argv[1:]
else:
    files = ["gym_two_bound_short.json"]

files = ["gym_default.json"] + files

spec = {}
for file in files:
    with open(file) as json_file:
        loaded = json.load(json_file)
        update_dict(spec, loaded)

start_time = datetime.datetime.now()
start_time_s = start_time.strftime("%Y-%m-%d_%H-%M-%S")
exp_name = files[-1][:-5] + "_" + format(start_time_s)
os.mkdir("./experiments/{}".format(exp_name))

pid = os.getpid()
py = psutil.Process(pid)

models = {
    "sac": SAC,
    "ddpg": DDPG,
    "ql": QLModel,
    "random": RandomModel
}

policies = {
    "sac": MlpPolicy,
    "ddpg":  DDPGMlpPolicy,
    "ql": None,
    "random": None
}

env_types = {
    'oligopoly': OligopolyEnv,
    'duopoly': MarketEnv
}
env_cls = env_types.get(spec.get("gym_type", "duopoly"))

with open("./experiments/{}/strategies.csv".format(exp_name), "w+") as strat_file, \
        open("./experiments/{}/strategies_early.csv".format(exp_name), "w+") as early_strat_file:
    def get_strategy(m, ref):
        data = []
        for p in range(1, 51):
            obs = np.asarray([[ref, p]])
            if spec["gym_type"] != "duopoly":
                obs = np.ones((1, len(spec["gym_params"]["comp_params"]))) * ref
                obs[0] = p
            data.append(orig_env.fix_action(m.predict(obs)[0][0][0], p))
        return data

    def plot_strategy(data, ref):
        plt.plot(range(1, 51), data, label="{}".format(ref))


    strat_writer = csv.writer(strat_file, delimiter=" ")
    early_strat_writer = csv.writer(early_strat_file, delimiter=" ")
    losses = []


    def callback(local_vars, global_vars):
        global losses
        if local_vars['step'] % 100 == 0 and 0 < local_vars['step'] <= 10000:
            strat = get_strategy(model, 30)
            early_strat_writer.writerow([exp_num, local_vars['step']] + strat)
            early_strat_file.flush()

        if local_vars['step'] % 1000 == 0 and local_vars['step'] > 0:
            print("Storing intermediate strategy for exp. {} step {}".format(exp_num, local_vars['step']))
            strat = get_strategy(model, 30)
            if spec["plot_strats"]:
                plot_strategy(strat, 30)
                plt.legend()
                plt.savefig("./experiments/{}/predictions_{}_{}.png".format(exp_name, exp_num, local_vars['step']))
                plt.close()

            strat_writer.writerow([exp_num, local_vars['step']] + strat)
            strat_file.flush()

        if 'losses' in local_vars:
            losses = local_vars['losses']
        return True


    with open("./experiments/{}/conf.json".format(exp_name), "w+") as f:
        # Specs change, our backup doesnt!
        json.dump(spec, f, indent=2)

    for exp_num in range(0, spec["num_experiments"]):
        with tf.Session() as sess:
            orig_env = env_cls(**spec["gym_params"])
            env = DummyVecEnv([lambda: orig_env])

            print("Start training exp. {}".format(exp_num))

            model = models.get(spec["model"])(policies.get(spec["model"]), env, **spec["model_params"])
            model.learn(total_timesteps=spec["train_timesteps"], log_interval=10, callback=callback)

            ep = 0

            orig_env.store_history("./experiments/{}/train_history_{}.json".format(exp_name, exp_num), losses=losses)
            print("Finish training exp. {}".format(exp_num))

            obs = env.reset()
            while ep < spec["eval_timesteps"]:
                ep += 1
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                env.render()

            orig_env.display("Final eval result exp. {}".format(exp_num) + ": {}/{}")
            orig_env.store_history("./experiments/{}/eval_history_{}.json".format(exp_name, exp_num), losses=None)

            memoryUse = py.memory_info()[0] / 2. ** 30
            print('Mem before cleanup: ', memoryUse)
            env.reset()
            del model
            del env
            del orig_env
            memoryUse = py.memory_info()[0] / 2. ** 30
            print('Mem after cleanup: ', memoryUse)

print("Done with all experiments, file handles closed.")
