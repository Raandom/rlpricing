import matplotlib.pyplot as plt
##############################################
# Configure script here
##############################################

files = [
    {
        "source": "./experiments/.../strategies.csv",
        "expnum": 6,
        "ep": 365000,
        "fname": "example_start.png",
    },
]

for spec in files:
    data = None
    with open(spec["source"], "r") as infile:

        i = 1
        for line in infile.readlines():
            content = [float(x) for x in line.split(" ")]
            exp_num = int(content[0])
            epi = int(content[1])
            if exp_num == spec["expnum"] and epi == spec["ep"]:
                data = content[2:]
                break
            i += 1
    if data:
        plt.plot(range(1, 51), data)
        plt.legend()
        axes = plt.gca()
        axes.set_xlabel("Competitor price")
        axes.set_ylabel("Response price")
        plt.savefig("./imageout/{}".format(spec["fname"]))
        plt.close()
        print("{} done".format(spec["fname"]))
    else:
        print("{} data not found".format(spec["fname"]))