import os

##############################################
# Configure script here
##############################################

# List of strategy files to merge
files = []

# Target dir to store resulting strategy file
target_dir = os.path.dirname(files[0])
target_file = os.path.join(target_dir, "strategies_merged.csv")

num_exp = -1
with open(target_file, "w+") as outfile:
    for file in files:
        with open(file, "r") as infile:
            for line in infile.readlines():
                if len(line) > 1:
                    line = line.split(" ")
                    if line[1] == "1000":
                        num_exp += 1
                    line[0] = str(num_exp)
                    line = " ".join(line)
                    outfile.write(line)
