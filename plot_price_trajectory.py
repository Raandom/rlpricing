import json
import matplotlib.pyplot as plt

##############################################
# Configure script here
##############################################

TARGET = ".../eval_history_0.json"
START = 5000
FRAME = 20
LABELS = [
    "SAC",
    "Two-bound 1-50",
    "Two-bound 15-45",
    "Fixed 19",
    "Random 1-50",
    "Mixed"
]

with open(TARGET) as f:
    data = json.load(f)

all_frames = []
for d in data:
    frames = []
    last = None
    for p in d["prices"]:
        if p[0] > START and p[0] <= START + FRAME:
            if len(frames) == 0:
                if last:
                    frames.append([0, last[1]])
                else:
                    frames.append([0, 1])
            p[0] = p[0] - START
            frames.append(p)
        if p[0] > START + FRAME:
            frames.append([FRAME, frames[-1][1]])
            break
        last = p
    all_frames.append(frames)

plt.figure(figsize=(8, 4), dpi=200)

for i, f in enumerate(all_frames):
    x = [t[0] + START for t in f]
    y = [t[1] for t in f]
    plt.step(x, y, label=LABELS[i], where="post")


axes = plt.gca()
axes.set_xlabel("Episode")
axes.set_ylabel("Price")
plt.legend()
plt.show()
print("Done")
