import os
import sys
lib_path = os.path.abspath(os.path.join('..', 'lib'))
sys.path.append(lib_path)

from icd9 import ICD9
import math
import gensim
import csv


class NearestNeighbor:

    def __init__(self, filename, decay=0, k=3, f=5, window=600, size=600):
        self._hit = self._miss = 0
        self._diags = set()
        self._uniq_events = set()

        # Parameters
        self._filename = filename
        self._decay = decay
        self._k = k
        self._f = f
        self._window = window
        self._size = size

        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                events = line.replace(",", "")[:-1].split(' ')
                self._uniq_events |= set(events)
                self._diags |= set([x for x in events if x.startswith('d_')])

        self._events_index = sorted(self._uniq_events)
        self._diags = list(self._diags)
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

    def train(self, filename):
        self._nn_mat = []
        self._prediction = []
        self._prediction_lists = []
        self._filename = filename
        self._events_mask = {}

        with open(filename) as f:
            sentences = [s[:-1].replace(",", "").split(' ') for s in f.readlines()]
            self._model = gensim.models.Word2Vec(sentences, sg=0, window=self._window,
                                                 size=self._size, min_count=1, workers=20)

        for diag in self._diags:
            self._events_mask[diag] = [0] * len(self._uniq_events)
            for e, d in self._model.most_similar(diag, topn=self._f):
                self._events_mask[diag][self._events_index.index(e)] = 1

        with open(filename) as f:
            for line in f:
                feed_index = line[0:line.rfind(" d_")].rfind(",")
                feed_events = line[0:feed_index].replace(",", "").split(" ")
                last_admission = line[feed_index:].replace("\n", "").replace(",", "").split(" ")
                actual = set([x for x in last_admission if x.startswith('d_')])

                seq_array = [0] * len(self._uniq_events)
                te = len(feed_events)
                for i, e in enumerate(feed_events):
                    seq_array[self._events_index.index(e)] += math.exp(self._decay*(i-te+1)/te)
                self._nn_mat.append(seq_array)

                result = [0] * len(self._diags)
                for diag in actual:
                    result[self._diags.index(diag)] = 1
                self._prediction.append(result)
                self._prediction_lists.append(actual)

    def predict(self, pred_seq):
        prediction = set()
        for diag in self._diags:
            dist = []
            for seq in self._nn_mat:
                dist.append(sum([m*(x-y)**2 for x, y, m in
                                 zip(seq, pred_seq, self._events_mask[diag])]))

            min_values = sorted(dist)
            count = 0
            for i in range(self._k):
                count += diag in self._prediction_lists[dist.index(min_values[i])]

            if count > self._k / 2.0:
                prediction.add(diag)

        return prediction

    def test(self, filename):
        total_test = 0
        with open(filename) as f:
            for line in f:
                feed_index = line[0:line.rfind(" d_")].rfind(",")
                feed_events = line[0:feed_index].replace(",", "").split(" ")
                last_admission = line[feed_index:].replace("\n", "").replace(",", "").split(" ")
                actual = set([x for x in last_admission if x.startswith('d_')])

                seq_array = [0] * len(self._uniq_events)
                te = len(feed_events)
                for i, e in enumerate(feed_events):
                    seq_array[self._events_index.index(e)] += math.exp(self._decay*(i-te+1)/te)

                prediction = self.predict(seq_array)
                print(prediction)
                print(actual)
                print("==========")

                for act in actual:
                    if act in prediction:
                        self._stats[act]["TP"] += 1
                    else:
                        self._stats[act]["FN"] += 1

                for pred in prediction - actual:
                    self._stats[pred]["FP"] += 1

                total_test += 1
                if len([x for x in actual if x in prediction]) > 0:
                    self._hit += 1
                else:
                    self._miss += 1

        for d in self._diags:
            self._stats[d]["TN"] = total_test - self._stats[d]["TP"] - \
                self._stats[d]["FN"] - self._stats[d]["FP"]

        return {"hit": self._hit, "miss": self._miss}

    def cross_validate(self, train_files, test_files):
        for i, train in enumerate(train_files):
            self.train(train)
            self.test(test_files[i])
            print(i, self.accuracy)

    @property
    def accuracy(self):
        return (1.0 * self._hit / (self._miss + self._hit))

    def report_accuracy(self):
        with open('../Results/accuracies.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.__class__.__name__, self.accuracy])

    def write_stats(self):
        with open('../Results/Stats/' + self.__class__.__name__ + '_k' + str(self._k) +
                  '_f' + str(self._f) + '_w'+str(self._window) + '_s' + str(self._size) +
                  '_d' + str(self._decay)+'.csv', 'w') as csvfile:
            header = ["Stat"]
            desc = ["Description"]
            spec = ["Specificity"]
            sens = ["Sensitivity"]
            acc = ["Accuracy"]
            tp = ["True Positives"]
            tn = ["True Negatives"]
            fp = ["False Positives"]
            fn = ["False Negatives"]
            for d in self._diags:
                spec.append(self._stats[d]["TP"]*1.0 /
                            (self._stats[d]["TP"] + self._stats[d]["FN"]))
                sens.append(self._stats[d]["TN"]*1.0 /
                            (self._stats[d]["FP"] + self._stats[d]["TN"]))
                header.append(d)
                desc.append(self._diag_to_desc[d])
                acc.append((self._stats[d]["TN"]*1.0 + self._stats[d]["TP"]) /
                           sum(self._stats[d].values())*1.0)
                tp.append(self._stats[d]["TP"])
                tn.append(self._stats[d]["TN"])
                fp.append(self._stats[d]["FP"])
                fn.append(self._stats[d]["FN"])

            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(desc)
            writer.writerow(spec)
            writer.writerow(sens)
            writer.writerow(acc)
            writer.writerow(tp)
            writer.writerow(tn)
            writer.writerow(fp)
            writer.writerow(fn)


if __name__ == '__main__':
    model = NearestNeighbor('../Data/mimic_train_cs_0', decay=5, k=5, f=8, window=600, size=600)
    train_files = []
    test_files = []
    for i in range(1):
        train_files.append('../Data/mimic_train_cs_'+str(i))
        test_files.append('../Data/mimic_test_cs_'+str(i))

    model.cross_validate(train_files, test_files)
    model.report_accuracy()
    model.write_stats()
    print(model.accuracy)
