import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import seaborn

N = 6

original = [0.702525618, 0.845864344, 0.921585423, 0.876446217, 0.87607845, 0.904038981]
balanced = [0.703906675, 0.836469726, 0.924487831, 0.883193988, 0.878495243, 0.886967934]
dynamic = [0.702994383, 0.845204987, 0.921338504, 0.875883499, 0.875435969, 0.903590325]

ind = .25 + np.arange(N)  # the x locations for the groups
width = 0.25       # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6), dpi=40)

u_data = ax.bar(ind, original, width, color='#ff834e')
b_data = ax.bar(ind + width, balanced, width, color='#4ed6ff')
d_data = ax.bar(ind + 2*width, balanced, width, color='#6eff6f')

# add some text for labels, title and axes ticks
aggx = ["Acute Renal Failure", "Heart failure", "Diabetes", "Hypothyroidism",
        "Asthma", "Sleep Disorders"]
ax.set_ylabel('AUC')
ax.set_xticks(ind + width + 0.125)
ax.set_xticklabels(aggx)

ax.legend((u_data[0], b_data[0], d_data), ('Original', 'Balanced', 'Dynamic Window'), loc=2)
plt.ylabel('AUC', fontsize=14, color='black')
# def autolabel(rects):
#     # attach some text labels
#     for rect in rects:
#         height = rect.get_height()
#         ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, '%d' % int(height),
#                 ha='center', va='bottom')
#
# autolabel(u_data)
# autolabel(b_data)

plt.savefig('balance_barchart.png')
