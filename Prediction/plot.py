import matplotlib.pyplot as plt
import random
import csv
import json


def plot(x, filters):
    x_values = {}
    y_values = {}

    with open('../Results/accuracies.csv', 'rU') as f:
        reader = csv.reader(f)
        for line in reader:
            props = json.loads(line[1])

            filtered_out = False
            for f in filters:
                if f in props:
                    if str(filters[f]) != str(props[f]):
                        filtered_out = True

            if filtered_out:
                continue

            label = "-".join(k[0]+":"+str(props[k]) for k in props
                             if k not in filters.keys() and k != x)
            if label in y_values:
                y_values[label].append(line[0])
                x_values[label].append(props[x])
            else:
                y_values[label] = [line[0]]
                x_values[label] = [props[x]]

    labels = x_values.keys()

    for label in sorted(labels):
        y = [vy for (vx, vy) in sorted(zip(x_values[label], y_values[label]))]
        plt.plot(sorted(x_values[label]), y,
                 color=("#%06x" % random.randint(0, 0xFFFFFF)), label=label)

    plt.title("Accuracy vs "+x)
    plt.xlabel(x, fontsize=14, color='black')
    plt.ylabel('Accuracy', fontsize=14, color='black')
    plt.legend(framealpha=0.5)
    filter_str = "(" + "-".join([k+"="+filters[k] for k in filters]) + ")"
    plt.savefig('../Results/Plots/'+filter_str+'_'+x+'.png')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot')
    parser.add_argument('-x', '--xaxis', action="store", type=str, required=True)
    parser.add_argument('-f', '--filters', nargs='+', action="store", type=str)
    args = parser.parse_args()
    filters = {}
    if args.filters is not None:
        filters = {filt.split(":")[0]: filt.split(":")[1] for filt in args.filters}
    plot(args.xaxis, filters)
