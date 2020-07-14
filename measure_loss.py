import json
import os
import numpy as np
import matplotlib.pyplot as plt


def measure_loss(dir):
    files = [os.path.join(dir, "train_history_{}.json".format(i)) for i in range(0, 10)]
    losses = []
    for f in files:
        with open(f) as file:
            cont = json.load(file)["losses"]
            losses.append(cont)
    return np.mean(losses, axis=0)

##############################################
# Configure script here
##############################################

# Source directories
sources = []

# Labels for each directory
labels = []


PLOT_COLORS = ["#CC4F1B", "#7167FF", "#80477B", "#14FFAD"]

losses_g = []
for dir in sources:
    losses_g.append(measure_loss(dir))

xes = [i * 128 for i in range(0, len(losses_g[0]))]
plt.figure(figsize=(8, 4), dpi=200)
for i, l in enumerate(losses_g):
    plt.plot(xes, l, label=labels[i], color=PLOT_COLORS[i % len(PLOT_COLORS)])

axes = plt.gca()
axes.set_ylim([0.0, 0.0005])
plt.legend()
plt.show()
