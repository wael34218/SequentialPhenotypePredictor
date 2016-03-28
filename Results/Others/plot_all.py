import matplotlib.pyplot as plt
import csv

plt.figure(figsize=(20, 5), dpi=120)
#fig, ax = plt.subplots(figsize=(16, 10), dpi=120)
colors = ['#A4036F', '#048BA8', '#16DB93', '#1FEA1A', '#F24E1C']
with open('all.csv', 'rU') as f:
    reader = csv.reader(f)
    x = []
    i = 0
    for line in reader:
        if len(x) == 0:
            x = line[1:]
            continue

        plt.plot(range(len(x)), line[1:], label=line[0], linewidth=1, color=colors[i])
        i += 1

plt.ylabel('AUC', fontsize=14, color='black')
plt.xticks(range(len(x)), [a[2:] for a in x], fontsize=11)
plt.xlim([0,len(x)-1])
plt.legend(framealpha=0.5, loc=4, fontsize=11)
# ax.set_xticklabels(x)
plt.savefig('all.png')
