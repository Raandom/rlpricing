import json
import numpy as np

import matplotlib.pyplot as plt

##############################################
# Configure script here
##############################################

# All directories/training history files to read data from
FILES = [e + "/train_history_0.json" for e in []]

data = []
for f in FILES:
    with open(f, "r") as f_o:
        data.append(json.load(f_o))

for episode in range(0, 1000000, 1000000):
    for perspective in ["a", "b"]:
        counter = "b" if perspective == "a" else "a"

        def is_training(x):
            if perspective == "a":
                return x % 100000 < 50000
            return x % 100000 >= 50000
        start = episode
        end = start + 1000000
        results = np.zeros((52, 52))
        for j in range(len(data)):
            en_price = 30 if start == 0 else data[j][counter]["prices"][start - 1]
            for i in range(start, end):
                reaction = data[j][perspective]["prices"][i]
                if not is_training(i):
                    results[round(reaction)][round(en_price)] += 1
                en_price = data[j][counter]["prices"][i]

        fig, ax = plt.subplots()
        training = "a" if (episode // 50000) % 2 == 0 else "b"
        plt.title("Reaction heatmap of comp. {}".format(perspective))
        plt.xlabel("Competitor price")
        plt.ylabel("Own reaction")
        im = ax.imshow(results, origin="lower", cmap="inferno")
        plt.show()
        plt.close()
