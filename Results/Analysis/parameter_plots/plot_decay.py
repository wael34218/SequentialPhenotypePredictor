import matplotlib.pyplot as plt
import seaborn
import csv

selected_diags = ["d_584", "d_428", "d_416", "d_486", "d_493", "d_327", "d_038"]
plt.figure(figsize=(10, 8), dpi=100)
with open('cbow_decay_mimic.csv', 'rU') as f:
    reader = csv.reader(f)
    x = []
    for line in reader:
        if len(x) == 0:
            x = line[2:]
            continue

        if line[0] in selected_diags:
            plt.plot(x, line[2:], label=line[1], linewidth=2)

plt.xlabel('Decay Factor', fontsize=14, color='black')
plt.ylabel('AUC', fontsize=14, color='black')
plt.legend(framealpha=0.5, loc=4, fontsize=12)
plt.savefig('cbow_decay.png')
