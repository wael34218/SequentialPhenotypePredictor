from random import randrange, shuffle
from collections import defaultdict

import os
from os import path

raw_dir = "../Data/seq_combined/"
balanced_raw_dir = "../Data/seq_combined_balanced/"

files = [f for f in os.listdir("../Data/seq_combined/")
         if path.isfile(path.join(raw_dir, f)) and not f.startswith(".")]

for f in files:
    total = 0
    diag_lines = defaultdict(lambda: [])
    diag_counts = defaultdict(lambda: 0)
    final_lines = []

    with open(path.join(raw_dir, f)) as old:
        for line in old:
            final_lines.append(line)
            total += 1
            diags = line.split("|")[0].split(",")
            for d in diags:
                diag_lines[d].append(line)
                diag_counts[d] += 1

    while min(diag_counts.values()) < int(total / 10):
        minimum = int(total / 10)
        d = min(diag_counts, key=diag_counts.get)
        for _ in range(minimum + 5 - diag_counts[d]):
            line = diag_lines[d][randrange(len(diag_lines[d]))]
            total += 1
            diags = line.split("|")[0].split(",")
            final_lines.append(line)
            for di in diags:
                diag_counts[di] += 1

    shuffle(final_lines)
    with open(path.join(balanced_raw_dir, f), 'w') as new:
        for line in final_lines:
            new.write(line)
