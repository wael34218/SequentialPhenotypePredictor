import os
import sys
lib_path = os.path.abspath(os.path.join('..', 'lib'))
sys.path.append(lib_path)

from icd9 import ICD9
import gensim
import math
import csv


class CbowSim:

    def __init__(self, filename):
        self._hit = self._miss = 0
        self._uniq_events = set()
        self._diags = set()
        self._filename = filename

        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                events = line[:-1].split(' ')
                self._uniq_events |= set(events)
                self._diags |= set([x for x in events if x.startswith('d_')])

        self._events_index = sorted(self._uniq_events)
        self._reset_stats()
        self._generate_icd9_lookup()

    def _reset_stats(self):
        self._stats = {}
        for diag in self._diags:
            self._stats[diag] = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    def _generate_icd9_lookup(self):
        self._diag_to_desc = {}
        tree = ICD9('../lib/icd9/codes.json')

        for d in self._diags:
            try:
                self._diag_to_desc[d] = tree.find(d[2:]).description
            except:
                if d[2:] == "285.9":
                    self._diag_to_desc[d] = "Anemia"
                elif d[2:] == "287.5":
                    self._diag_to_desc[d] = "Thrombocytopenia"
                elif d[2:] == "285.1":
                    self._diag_to_desc[d] = "Acute posthemorrhagic anemia"
                else:
                    self._diag_to_desc[d] = "Not Found"

    def train(self, filename, window=600, size=600):
        self._sim_mat = {}
        self._filename = filename
        self._window = window
        self._size = size
        with open(filename) as f:
            sentences = [s[:-1].split(' ') for s in f.readlines()]
            self._model = gensim.models.Word2Vec(sentences, sg=0, window=window, size=size,
                                                 min_count=1, workers=20)

        for diag in self._diags:
            words = self._model.most_similar(diag, topn=len(self._uniq_events))
            sim_array = [0] * len(self._uniq_events)
            sim_array[self._events_index.index(diag)] = 1
            for event, distance in words:
                sim_array[self._events_index.index(event)] = distance
            self._sim_mat[diag] = sim_array

    def test(self, filename, decay_coeff=5):
        total_test = 0
        self._decay_coeff = decay_coeff
        with open(filename) as f:
            for line in f:
                total_test += 1
                feed_index = line[0:line.rfind(" d_")].rfind(",")
                feed_events = line[0:feed_index].replace(",", "").split(" ")
                last_admission = line[feed_index:].replace("\n", "").replace(",", "").split(" ")
                actual = set([x for x in last_admission if x.startswith('d_')])
                test_array = [0] * len(self._uniq_events)

                te = len(feed_events)
                for i, e in enumerate(feed_events):
                    test_array[self._events_index.index(e)] += math.exp(decay_coeff*(i-te+1)/te)

                result = {}
                for diag in self._sim_mat:
                    result[sum([x*y for x, y in zip(test_array, self._sim_mat[diag])])] = diag

                distances = sorted(result.keys(), reverse=True)[:5]
                prediction = set([result[x] for x in distances])
                # prediction |= set([x for x in feed_events if x.startswith('d_')])

                for act in actual:
                    if act in prediction:
                        self._stats[act]["TP"] += 1
                    else:
                        self._stats[act]["FN"] += 1

                for pred in prediction - actual:
                    self._stats[pred]["FP"] += 1

                if len([x for x in actual if x in prediction]) > 0:
                    self._hit += 1
                else:
                    self._miss += 1

        for d in self._diags:
            self._stats[d]["TN"] = total_test - self._stats[d]["TP"] - \
                self._stats[d]["FN"] - self._stats[d]["FP"]

        return {"hit": self._hit, "miss": self._miss}

    def cross_validate(self, train_files, test_files, window, size, decay):
        for i, train in enumerate(train_files):
            self.train(train, window, size)
            self.test(test_files[i], decay)

    def accuracy(self):
        return (1.0 * self._hit / (self._miss + self._hit))

    def report_accuracy(self):
        with open('../Results/accuracies.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['cbowsim', self._size, self._window,
                             self._decay_coeff, self.accuracy()])

    def write_stats(self):
        with open('../Results/Stats/' + self.__class__.__name__ + '_'+str(self._window) +
                  '_'+str(self._size)+'_'+str(self._decay_coeff)+'.csv', 'w') as csvfile:
            header = ["Stat"]
            desc = ["Description"]
            spec = ["Specificity"]
            sens = ["Sensitivity"]
            for d in self._diags:
                spec.append(self._stats[d]["TP"]*1.0 /
                            (self._stats[d]["TP"] + self._stats[d]["FN"]))
                sens.append(self._stats[d]["TN"]*1.0 /
                            (self._stats[d]["FP"] + self._stats[d]["TN"]))
                header.append(d)
                desc.append(self._diag_to_desc[d])

            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(desc)
            writer.writerow(spec)
            writer.writerow(sens)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CBOW Similarity')
    parser.add_argument('--window', action="store", default=600, type=int)
    parser.add_argument('--size', action="store", default=600, type=int)
    parser.add_argument('--decay', action="store", default=5, type=float)
    args = parser.parse_args()

    train_files = []
    test_files = []
    model = CbowSim('../Data/mimic_train_0')

    for i in range(10):
        train_files.append('../Data/mimic_train_'+str(i))
        test_files.append('../Data/mimic_test_'+str(i))

    model.cross_validate(train_files, test_files, args.window, args.size, args.decay)
    model.report_accuracy()
    model.write_stats()
