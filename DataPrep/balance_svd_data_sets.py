from random import randrange, shuffle

import os
from os import path

svd_dir = "../Data/svd/"
balanced_svd_dir = "../Data/svd_balanced/"

files = [f for f in os.listdir("../Data/svd/")
         if path.isfile(path.join(svd_dir, f)) and not f.startswith(".")]

for f in files:
    positives = []
    negatives = []
    with open(path.join(svd_dir, f)) as old:
        for line in old.readlines():
            if line.startswith("1"):
                positives.append(line)
            else:
                negatives.append(line)

    if len(negatives) > len(positives):
        original_len = len(positives)
        for _ in range(original_len, len(negatives)):
            positives.append(positives[randrange(original_len)])
    else:
        original_len = len(negatives)
        for _ in range(original_len, len(positives)):
            negatives.append(negatives[randrange(original_len)])

    all_data = positives + negatives
    shuffle(all_data)
    with open(path.join(balanced_svd_dir, f), 'w') as new:
        for line in all_data:
            new.write(line)
