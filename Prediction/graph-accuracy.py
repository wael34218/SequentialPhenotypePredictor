import matplotlib.pyplot as plt
import random

x_values = {}
y_values = {}
last_size = None

with open('accuracy.txt') as f:
    lines = f.readlines()
    for line in lines:
        if line.find("=") > -1:
            continue
        elif line.find("Window") < 0:
            if last_size in y_values:
                y_values[last_size].append(float(line))
            else:
                y_values[last_size] = [float(line)]
        else:
            last_size = line.split("-")[0].split(" ")[1]
            last_window = line.split("-")[1].split(" ")[1]
            if last_size in x_values:
                x_values[last_size].append(float(last_window))
            else:
                x_values[last_size] = [float(last_window)]

windows = x_values.keys()

for line in sorted(windows):
    plt.plot(x_values[line], y_values[line], color=("#%06x" % random.randint(0, 0xFFFFFF)), label=line)

plt.title("Accuracy using different window and size parameters")
plt.xlabel('Window', fontsize=14, color='black')
plt.ylabel('Accuracy', fontsize=14, color='black')
plt.legend(framealpha=0.5)
plt.show()
