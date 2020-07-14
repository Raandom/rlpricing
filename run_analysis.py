import csv
import json
import subprocess
import sys
import os
import warnings
from shutil import copyfile
import numpy as np
from matplotlib import pyplot as plt

PLOT_PERF_ANALYSIS = True
COMPUTE_STRAT_COMPLEXITY = False
NO_CACHE = False
# Write our source file to the correct position for ampl to use
def read_file(target_file):
    chunk = 100 if target_file.endswith("_early.csv") else 1000

    dir = os.path.dirname(target_file)
    conf_file = os.path.join(dir, "conf.json")
    with open(conf_file, "r") as conf:
        cont = json.load(conf)
        num_exp = cont["num_experiments"]
        max_epi = cont["train_timesteps"] // chunk - 1

    if target_file.endswith("_merged.csv"):
        num_exp = 10

    if target_file.endswith("_early.csv"):
        max_epi = 100

    print("Copying source file {}".format(target_file))
    if os.path.isfile("infile.csv"):
        os.remove("infile.csv")
    copyfile(target_file, "infile.csv")
    print("Done")

    print("Transforming infile.csv to input.txt")
    if os.path.isfile("input.txt"):
        os.remove("input.txt")
    strategy_store = np.zeros(shape=(num_exp, max_epi, 50))
    cache_file = target_file + ".npy"
    with open("infile.csv", "r") as infile, open("input.txt", "w+") as outfile:
        outfile.writelines([
            "param dl := {};\n".format(num_exp),
            "param epi := {};\n".format(max_epi * chunk),
            "param chunk := {};\n".format(chunk),
            "param input: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 :=\n",
        ])

        i = 1
        for line in infile.readlines():
            outfile.write(str(i) + " " + line)
            content = [float(x) for x in line.split(" ")]
            exp_num = int(content[0])
            epi = int(content[1]/chunk) - 1
            strategy_store[exp_num][epi] = content[2:]
            i += 1

        outfile.write(";")

        print("Done")
    # Launch ampl, import command
    if PLOT_PERF_ANALYSIS:
        if os.path.isfile(cache_file) and not NO_CACHE:
            print("Cache file found, using it")
            result_out = np.load(cache_file)
            max_out = np.load(os.path.join(dir, "maxout.npy"))[0]
        else:
            print("Launching ampl")
            proc = subprocess.Popen(["./ampl"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            out = proc.communicate(input=b'include "evaluate_strategy.txt"; \n')
            print("Done")
            proc.kill()
            print("Results")
            out = out[0].decode()
            out = out.split("\n")
            last_solver_output = next(i for i in reversed(range(len(out))) if out[i].startswith("MINOS"))
            end_pos = last_solver_output - len(out)
            result_out = out[(end_pos - max_epi):end_pos]
            max_out = float(out[-3][12:])
            print("\n".join(result_out))
            print("Max val {}".format(max_out))
            result_out = np.array([[float(el) for el in line.split(" ") if el != ""] for line in result_out])
            np.save(cache_file, result_out)
            np.save(os.path.join(dir, "maxout.npy"), np.asarray([max_out]))
        steps = result_out[:, 0]
        result_out = result_out[:, 1:]

        print("Done loading source file {}".format(target_file))
        return result_out, result_out / max_out, max_out, steps, strategy_store
    else:
        print("Skipping AMPL")
        return None, None, None, None, strategy_store

PLOT_COLORS = ["#CC4F1B", "#7167FF", "#80477B", "#14FFAD"]
PLOT_SHADOW_COLORS = ['#FF9848', "#B8B3FF", "#FF8FF6", "#14FFAD"]
PLOT_SHADOW_BORDER_COLORS = PLOT_COLORS


def plot_relative_performance(steps, outputs, labels, plot_shadows=True, filename=None):

    lens = [len(s) for s in steps]
    shortest = np.argmin(lens)
    steps = steps[shortest]
    outputs = [o[:len(steps)] for o in outputs]

    plt.figure(figsize=(8, 4), dpi=200)
    for i, o in enumerate(outputs):
        mean_results = np.median(o, axis=1)
        upper_quantile = np.quantile(o, .8, axis=1)
        lower_quantile = np.quantile(o, .2, axis=1)
        plt.plot(steps, mean_results, 'k', color=PLOT_COLORS[i % len(PLOT_COLORS)], label="{}".format(labels[i]))
        if plot_shadows:
            plt.fill_between(steps, upper_quantile, lower_quantile,
                alpha=0.5, edgecolor=PLOT_SHADOW_BORDER_COLORS[i % len(PLOT_SHADOW_BORDER_COLORS)], facecolor=PLOT_SHADOW_COLORS[i % len(PLOT_SHADOW_COLORS)])

    axes = plt.gca()
    axes.set_ylim([0, 1])
    axes.set_xlabel("Episode")
    axes.set_ylabel("Normalized strategy value")
    if len(outputs) > 1:
        plt.legend()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


print("Loading starts")
overall_steps = []
overall_max_out = None
results = []
relative_results = []
strategy_results = []

##############################################
# Configure script here
##############################################

# Strategy files to load
files = []
for f in files:
    result_out, relative_out, max_out, steps, strategies = read_file(f)
    results.append(result_out)
    relative_results.append(relative_out)
    strategy_results.append(strategies)
    overall_steps.append(steps)

# Tuples with indices that should be plotted
combos = []

# Labels for each tuple
combo_labels = []

# Filenames to store resulting plots
combo_filenames = []
combo_filenames = [os.path.join("./imageout/", f) for f in combo_filenames]

# Labels for the complexity plot
complexity_labels = []
if PLOT_PERF_ANALYSIS:
    print("Analyze perf history")
    for i, c in enumerate(combos):
        plot_relative_performance([overall_steps[idx] for idx in c], [relative_results[idx] for idx in c], combo_labels[i], filename=combo_filenames[i])
    print("Finally done, writing results")

    for i, res in enumerate(relative_results):
        opt = np.max(res)
        amax = np.unravel_index(np.argmax(res), res.shape)

        print("Run {}, max perf of {} reached in {}".format(files[i], opt, str(amax)))

if COMPUTE_STRAT_COMPLEXITY:
    print("Analyze strategy complexity")
    strat_complexity_stat = np.zeros(shape=(len(strategy_results),) + strategy_results[0].shape[:-1])
    for i in range(len(strategy_results)):
        for num_exp in range(len(strategy_results[i])):
            for epi in range(len(strategy_results[i][num_exp])):
                strat_complexity_stat[i][num_exp][epi] = len(np.unique(strategy_results[i][num_exp][epi]))

    y_range = list(range(1, 11))
    strat_complexity_counter = np.zeros((len(strategy_results), 10))
    for i in range(len(strat_complexity_stat)):
        result = np.unique(strat_complexity_stat[i], return_counts=True)
        for complexity, count in zip(*result):
            complexity = int(complexity) - 1
            if complexity <= len(strat_complexity_counter) - 2:
                strat_complexity_counter[i][complexity] = count
            else:
                strat_complexity_counter[i][-1] += count
        plt.plot(y_range, strat_complexity_counter[i], label=complexity_labels[i])

    plt.legend()
    plt.show()

    np.savetxt("./remotexperiments/complexity_analysis.csv", np.concatenate([np.asarray([complexity_labels], dtype=np.str).T, strat_complexity_counter], axis=1), fmt="%s", delimiter=",")

print("Donedonedone")