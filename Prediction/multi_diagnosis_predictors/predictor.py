import os
import sys
lib_path = os.path.abspath(os.path.join('..','..', 'lib'))
sys.path.append(lib_path)

from icd9 import ICD9
from sklearn import metrics
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

        self._nevents = len(self._uniq_events)
        self._events_index = sorted(self._uniq_events)
        self._reset_stats()
        self._generate_icd9_lookup()
        self._diags = list(self._diags)
        self._auc_enabled = False

    def _reset_stats(self):
        self._stats = {}
        self._true_vals = {}
        self._pred_vals = {}
        self._total_test = 0
        self._total_predictions = 0
        for diag in self._diags:
            self._stats[diag] = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
            self._true_vals[diag] = []
            self._pred_vals[diag] = []

    def _generate_icd9_lookup(self):
        self._diag_to_desc = {}
        tree = ICD9('../../lib/icd9/codes.json')

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
        self._total_predictions += len(prediction)
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
    def prediction_per_patient(self):
        return (1.0 * self._total_predictions / (self._miss + self._hit))

    @property
    def accuracy(self):
        return (1.0 * self._hit / (self._miss + self._hit))

    @property
    def csv_name(self):
        fname = self.__class__.__name__
        for k in sorted(self._props):
            fname += "_" + k[:2] + str(self._props[k])
        fname += ".csv"
        return fname

    def report_accuracy(self, calculate_true_negatives=True):
        if calculate_true_negatives:
            self._calculate_true_negatives()
        with open('../Results/accuracies.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            props = {k: self._props[k] for k in self._props}
            props["model"] = self.__class__.__name__
            writer.writerow([self.accuracy, json.dumps(props, sort_keys=True),
                             self.prediction_per_patient])

    def write_stats(self, calculate_true_negatives=True):
        if calculate_true_negatives:
            self._calculate_true_negatives()
        with open('../Results/Stats/' + self.csv_name, 'w') as csvfile:
            writer = csv.writer(csvfile)
            header = ["Diagnosis", "Description", "AUC", "F-Score", "Specificity", "Sensitivity",
                      "Accuracy", "True Positives", "True Negatives", "False Positives",
                      "False Negatives"]
            writer.writerow(header)
            for d in sorted(self._diags):
                # print(d, self._stats[d])
                row = []
                row.append(d)
                row.append(self._diag_to_desc[d])
                row.append(self._d_auc(d))
                row.append(self._d_fscore(d))
                row.append(self._d_specificity(d))
                row.append(self._d_accuracy(d))
                row.append(self._stats[d]["TP"])
                row.append(self._stats[d]["TN"])
                row.append(self._stats[d]["FP"])
                row.append(self._stats[d]["FN"])
                writer.writerow(row)

    def _d_auc(self, d):
        if self._auc_enabled:
            return (metrics.roc_auc_score(self._true_vals[d], self._pred_vals[d]))
        else:
            return "NA"

    def _d_specificity(self, d):
        if self._stats[d]["TP"] + self._stats[d]["FN"] == 0:
            return (self._stats[d]["TP"] / 1.0)
        else:
            return (self._stats[d]["TP"]*1.0 / (self._stats[d]["TP"] + self._stats[d]["FN"]))

    def _d_sensitivity(self, d):
        if self._stats[d]["FP"] + self._stats[d]["TN"] == 0:
            return (self._stats[d]["TN"] / 1.0)
        else:
            return (self._stats[d]["TN"]*1.0 / (self._stats[d]["FP"] + self._stats[d]["TN"]))

    def _d_accuracy(self, d):
        return (self._stats[d]["TN"]*1.0 + self._stats[d]["TP"]) / sum(self._stats[d].values())*1.0

    def _d_precision(self, d):
        if self._stats[d]["FP"] + self._stats[d]["TP"] == 0:
            return (self._stats[d]["TP"] / 1.0)
        else:
            return (self._stats[d]["TP"]*1.0 / (self._stats[d]["TP"] + self._stats[d]["FP"]))

    def _d_fscore(self, d):
        return ((2 * self._d_precision(d) * self._d_sensitivity(d)) /
                (self._d_precision(d) + self._d_sensitivity(d)))
