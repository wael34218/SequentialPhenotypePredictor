import os
import sys
lib_path = os.path.abspath(os.path.join('..', 'lib'))
sys.path.append(lib_path)

from icd9 import ICD9
import csv
import json


class Predictor(object):

    def __init__(self, filename):
        self._hit = self._miss = 0
        self._uniq_events = set()
        self._diags = set()
        self._filename = filename

        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                events = line[:-1].replace(",", "").split(' ')
                self._uniq_events |= set(events)
                self._diags |= set([x for x in events if x.startswith('d_')])

        self._events_index = sorted(self._uniq_events)
        self._reset_stats()
        self._generate_icd9_lookup()
        self._diags = list(self._diags)

    def _reset_stats(self):
        self._stats = {}
        self._total_test = 0
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

    def stat_prediction(self, prediction, actual):
        for act in actual:
            if act in prediction:
                self._stats[act]["TP"] += 1
            else:
                self._stats[act]["FN"] += 1

        for pred in prediction - actual:
            self._stats[pred]["FP"] += 1

        self._total_test += 1
        if len([x for x in actual if x in prediction]) > 0:
            self._hit += 1
        else:
            self._miss += 1

    def _calculate_true_negatives(self):
        for d in self._diags:
            self._stats[d]["TN"] = self._total_test - self._stats[d]["TP"] - \
                self._stats[d]["FN"] - self._stats[d]["FP"]

    def cross_validate(self, train_files, test_files):
        self._reset_stats()
        for i, train in enumerate(train_files):
            self.train(train_files[i])
            self.test(test_files[i])

    @property
    def accuracy(self):
        return (1.0 * self._hit / (self._miss + self._hit))

    @property
    def csv_name(self):
        fname = self.__class__.__name__
        for k in sorted(self._props):
            fname += "_" + k[:1] + str(self._props[k])
        fname += "_2.csv"
        return fname

    def report_accuracy(self):
        self._calculate_true_negatives()
        with open('../Results/accuracies.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            props = {k: self._props[k] for k in self._props}
            props["model"] = self.__class__.__name__
            writer.writerow([self.accuracy, json.dumps(props, sort_keys=True)])

    def write_stats(self):
        with open('../Results/Stats/' + self.csv_name, 'w') as csvfile:
            writer = csv.writer(csvfile)
            header = ["Diagnosis", "Description", "Specificity", "Sensitivity", "Accuracy",
                      "True Positives", "True Negatives", "False Positives", "False Negatives"]
            writer.writerow(header)
            for d in sorted(self._diags):
                row = []
                row.append(d)
                row.append(self._diag_to_desc[d])
                row.append(self._stats[d]["TP"]*1.0 / (self._stats[d]["TP"] + self._stats[d]["FN"]))
                row.append(self._stats[d]["TN"]*1.0 / (self._stats[d]["FP"] + self._stats[d]["TN"]))
                row.append((self._stats[d]["TN"]*1.0 + self._stats[d]["TP"]) /
                           sum(self._stats[d].values())*1.0)
                row.append(self._stats[d]["TP"])
                row.append(self._stats[d]["TN"])
                row.append(self._stats[d]["FP"])
                row.append(self._stats[d]["FN"])
                writer.writerow(row)
