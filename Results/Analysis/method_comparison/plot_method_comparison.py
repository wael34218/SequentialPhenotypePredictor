import matplotlib.pyplot as plt
import seaborn
import csv

plt.figure(figsize=(20, 5), dpi=120)
# fig, ax = plt.subplots(figsize=(16, 10), dpi=120)
colors = ['#34D421', '#E45241', '#111111', '#5463F2']
shapes = ['--', 'o', 'd', '^']
diags = ["d_244", "d_250", "d_272", "d_276", "d_285", "d_300", "d_311", "d_338", "d_401", "d_414",
         "d_427", "d_428", "d_486", "d_491", "d_493", "d_496", "d_518", "d_530", "d_564", "d_571",
         "d_572", "d_577", "d_578", "d_584", "d_585", "d_599", "d_600", "d_682", "d_707", "d_715"]

dataset = 'ucsd'
with open(dataset+'_methods.csv', 'rU') as f:
    reader = csv.reader(f)
    header = reader.next()

    graphs = {h: [] for h in header if h}
    index = {h: i for i, h in enumerate(header) if h}
    for line in reader:
        if line[0] in diags:
            for g in graphs:
                graphs[g].append(line[index[g]])

symbols = list(map(chr,range(65,91))) + list(map(chr,range(97,101)))

for i, g in enumerate(["Ttf-idf", "PDCS", "CF", "PDPS"]):
    plt.plot(range(len(graphs[g])), graphs[g], shapes[i], label=g, linewidth=1, color=colors[i])

plt.ylabel('AUC', fontsize=14, color='black')
# plt.xlabel('Diagnosis ICD9 level 3', fontsize=14, color='black')
plt.xticks(range(len(diags)), symbols, fontsize=11)
plt.xlim([-.25, len(diags)-.75])
plt.legend(framealpha=0.5, loc=3, fontsize=11)
# ax.set_xticklabels(x)
plt.savefig(dataset+'_comparison.png')
