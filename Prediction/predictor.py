import os
import sys
lib_path = os.path.abspath(os.path.join('..', 'lib'))
sys.path.append(lib_path)

from icd9 import ICD9
import csv


class Predictor:

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

    def calculate_true_negatives(self):
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
        for k in self._props:
            fname += "_" + k[:1] + str(self._props[k])
        fname += ".csv"
        return fname

    @property
    def row_values(self):
        rows = [self.__class__.__name__, self.accuracy]
        for k in self._props:
            rows.append(self._props[k])
        return rows

    def report_accuracy(self):
        with open('../Results/accuracies.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.row_values)

    def write_stats(self):
        with open('../Results/Stats/' + self.csv_name, 'w') as csvfile:
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
