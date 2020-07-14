import csv
import datetime
import json
import os
import sys
import psutil

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines import SAC, DDPG
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.sac import MlpPolicy
from stable_baselines.ddpg import MlpPolicy as DDPGMlpPolicy

from gymql.strategy import QLModel
from simtools import update, store_history
from environments.duopoly import DoubleMarketEnv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if len(sys.argv) > 1:
    file = " ".join(sys.argv[1:])
else:
    file = "double_gym_default.json"

with open(file) as json_file, open("double_gym_default.json") as default_spec:
    spec = json.load(default_spec)
    update(spec, json.load(json_file))

start_time = datetime.datetime.now()
start_time_s = start_time.strftime("%Y-%m-%d_%H-%M-%S")
exp_name = file[:-5] + "_" + format(start_time_s)
os.mkdir("./experiments/{}".format(exp_name))

pid = os.getpid()
py = psutil.Process(pid)

models = {
    "sac": SAC,
    "ddpg": DDPG,
    "ql": QLModel
}

policies = {
    SAC: MlpPolicy,
    DDPG:  DDPGMlpPolicy,
    QLModel: None
}

a_params = spec.get("model_params")
b_params = spec.get("model_b_params", a_params)

a_cls = models.get(spec.get("model"))
b_cls = models.get(spec.get("model_b"))

a_policy = policies.get(a_cls)
b_policy = policies.get(b_cls)

with open("./experiments/{}/strategies_a.csv".format(exp_name), "w+") as strat_file_a, open("./experiments/{}/strategies_b.csv".format(exp_name), "w+") as strat_file_b:
    def get_strategy(m, ref, a):
        data = []
        for p in range(1, 51):
            data.append(main_env.fix_action(m.predict(np.asarray([[ref, p]]))[0][0][0], a, p))
        return data

    def plot_strategy(data, ref):
        plt.plot(range(1, 51), data, label="{}".format(ref))


    strat_writer_a = csv.writer(strat_file_a, delimiter=" ")
    strat_writer_b = csv.writer(strat_file_b, delimiter=" ")


    def callback(local_vars, global_vars):
        if local_vars['step'] % 1000 == 0:
            print("Storing intermediate strategy for A exp. {} epoch {} step {}".format(exp_num, train_epoch, local_vars['step']))
            strat = get_strategy(model_a, 30, True)
            if spec["plot_strats"]:
                plot_strategy(strat, 30)
                plt.legend()
                plt.savefig("./experiments/{}/predictions_a_{}_{}_{}.png".format(exp_name, exp_num, train_epoch, local_vars['step']))
                plt.close()

            strat_writer_a.writerow([exp_num, local_vars['step']] + strat)
            strat_file_a.flush()
        return True

    def callback_b(local_vars, global_vars):
        if local_vars['step'] % 1000 == 0:
            print("Storing intermediate strategy for B exp. {} epoch {} step {}".format(exp_num, train_epoch, local_vars['step']))
            strat = get_strategy(model_b, 30, False)
            if spec["plot_strats"]:
                plot_strategy(strat, 30)
                plt.legend()
                plt.savefig("./experiments/{}/predictions_b_{}_{}_{}.png".format(exp_name, exp_num, train_epoch, local_vars['step']))
                plt.close()

            strat_writer_b.writerow([exp_num, local_vars['step']] + strat)
            strat_file_b.flush()
        return True


    with open("./experiments/{}/conf.json".format(exp_name), "w+") as f:
        # Specs change, our backup doesnt!
        json.dump(spec, f, indent=2)

    for exp_num in range(0, spec["num_experiments"]):
        main_env = DoubleMarketEnv(**spec["gym_params"])
        a_orig_env = main_env.get_env(True)
        b_orig_env = main_env.get_env(False)
        a_env = DummyVecEnv([lambda: a_orig_env])
        b_env = DummyVecEnv([lambda: b_orig_env])

        print("Start training exp. {}".format(exp_num))

        model_a = a_cls(a_policy, a_env, **a_params)
        model_b = b_cls(b_policy, b_env, **b_params)

        main_env.register_comp(model_a, True)
        main_env.register_comp(model_b, False)

        for train_epoch in range(0, spec["train_epochs"]):
            model_a.learn(total_timesteps=spec["train_timesteps"], log_interval=10, callback=callback)
            model_b.learn(total_timesteps=spec["train_timesteps"], log_interval=10, callback=callback_b)

        ep = 0

        store_history(
            "./experiments/{}/train_history_{}.json".format(exp_name, exp_num),
            main_env.overall_profits_a, main_env.overall_profits_b,
            main_env.hist_a, main_env.hist_b
        )

        main_env.finish_training()

        print("Finish training exp. {}".format(exp_num))

        a_orig_env.reset()
        while ep < spec["eval_timesteps"]:
            ep += 1
            action, _states = model_a.predict([[main_env.ref_p, main_env.price_b]])
            profits_a, profits_b = main_env.half_step(True, action[0][0], True)

            action, _state = model_b.predict([[main_env.ref_p, main_env.price_a]])
            profits_a_2, profits_b_2 = main_env.half_step(False, action[0][0], False)

            main_env.overall_profits_a += profits_a + profits_a_2
            main_env.overall_profits_b += profits_b + profits_b_2

        print("Final eval result exp. {}: {}/{}".format(exp_num, main_env.overall_profits_a, main_env.overall_profits_b))
        store_history(
            "./experiments/{}/eval_history_{}.json".format(exp_name, exp_num),
            main_env.overall_profits_a, main_env.overall_profits_b,
            main_env.hist_a, main_env.hist_b
        )

        memoryUse = py.memory_info()[0] / 2. ** 30
        print('Mem before cleanup: ', memoryUse)
        main_env.finish_training()
        del model_a
        del model_b
        del a_env
        del b_env
        del a_orig_env
        del b_orig_env
        del main_env
        memoryUse = py.memory_info()[0] / 2. ** 30
        print('Mem after cleanup: ', memoryUse)

print("Done with all experiments, file handles closed.")
