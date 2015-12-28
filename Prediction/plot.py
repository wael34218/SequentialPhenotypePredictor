import matplotlib.pyplot as plt
import random
import csv

accuracy_col = {'model': 0, 'size': 1, 'window': 2, 'decay': 3, 'accuracy': 4}


def plot(model, x):
    x_values = {}
    y_values = {}

    label_attrs = [k for k in accuracy_col.keys() if k not in [x, 'model', 'accuracy']]
    with open('../Results/accuracies.csv') as f:
        reader = csv.reader(f)
        for line in reader:
            if line[accuracy_col['model']] != model:
                continue

            label = "-".join([l+str(line[accuracy_col[l]]) for l in label_attrs])
            if label in y_values:
                y_values[label].append(line[accuracy_col["accuracy"]])
                x_values[label].append(line[accuracy_col[x]])
            else:
                y_values[label] = [line[accuracy_col["accuracy"]]]
                x_values[label] = [line[accuracy_col[x]]]

    labels = x_values.keys()

    for label in sorted(labels):
        plt.plot(x_values[label], y_values[label],
                 color=("#%06x" % random.randint(0, 0xFFFFFF)), label=label)

    plt.title("Accuracy vs "+x)
    plt.xlabel(x, fontsize=14, color='black')
    plt.ylabel('Accuracy', fontsize=14, color='black')
    plt.legend(framealpha=0.5)
    plt.savefig('../Results/Plots/'+model+'_'+x+'.png')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot')
    parser.add_argument('--x', action="store", type=str)
    parser.add_argument('--model', action="store", type=str)
    args = parser.parse_args()
    plot(args.model, args.x)
