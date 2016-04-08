import os
import sys
lib_path = os.path.abspath(os.path.join('..', 'lib'))
sys.path.append(lib_path)

from icd9 import ICD9
from sklearn import metrics
import csv
import json
from collections import defaultdict
import gensim
import operator
import matplotlib.pyplot as plt
import pickle
import math

from chao_word2vec.word2vec import Word2Vec

# import warnings
# warnings.filterwarnings("error")


class BinaryPredictor(object):

    def __init__(self, filename):
        self._hit = self._miss = 0
        self._uniq_events = set()
        self._diags = set()
        self._filename = filename

        with open(filename) as f:
            line = f.readline()
            self._uniq_events = set(line.split())
            line = f.readline()
            self._diags = set(line.split())

        self._nevents = len(self._uniq_events)
        self._events_index = sorted(self._uniq_events)
        self._reset_stats()
        self._generate_icd9_lookup()
        self._diags = list(self._diags)

    def _reset_stats(self):
        self._stats = {}
        self._true_vals = {}
        self._pred_vals = {}
        self._true_test = {}
        self._pred_test = {}
        self._total_predictions = 0
        for diag in self._diags:
            self._stats[diag] = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
            self._true_vals[diag] = []
            self._pred_vals[diag] = []
            self._true_test[diag] = []
            self._pred_test[diag] = []

    def _generate_icd9_lookup(self):
        self._diag_to_desc = {}
        tree = ICD9('../lib/icd9/codes.json')

        for d in self._diags:
            try:
                self._diag_to_desc[d] = tree.find(d[2:]).description
            except:
                if d[2:] == "008":
                    self._diag_to_desc[d] = "Intestinal infections due to other organisms"
                elif d[2:] == "280":
                    self._diag_to_desc[d] = "Iron deficiency anemias"
                elif d[2:] == "284":
                    self._diag_to_desc[d] = "Aplastic anemia and other bone marrow failure syndrome"
                elif d[2:] == "285":
                    self._diag_to_desc[d] = "Other and unspecified anemias"
                elif d[2:] == "286":
                    self._diag_to_desc[d] = "Coagulation defects"
                elif d[2:] == "287":
                    self._diag_to_desc[d] = "Purpura and other hemorrhagic conditions"
                elif d[2:] == "288":
                    self._diag_to_desc[d] = "Diseases of white blood cells"
                else:
                    self._diag_to_desc[d] = "Not Found"

    def lookup_diagnosis(self, diag):
        return self._diag_to_desc[diag]

    def base_train(self, filename, skipgram=0):
        '''
        This function trains the sequences on word2vec exculding stopwords and calculates prior
        probabilities. These 2 functions jumbled into one for efficiency.
        '''
        self._filename = filename
        self._prior = {}
        self._model = None

        diag_totals = defaultdict(lambda: 0)
        diag_joined = defaultdict(lambda: 0)
        sentences = []
        self.seq_count = 0

        with open(filename) as f:
            for s in f:
                self.seq_count += 1
                sentences.append(s.split("|")[2].split(" ") +
                                 s.split("|")[3].replace("\n", "").split(" "))
                next_diags = s.split("|")[0].split(",")
                prev_diags = [e for e in s.split("|")[2].split(" ") if e.startswith("d_")]
                for d in prev_diags:
                    diag_totals[d] += 1
                    if d in next_diags:
                        diag_joined[d] += 1

        for d in diag_totals:
            self._prior[d] = diag_joined[d] * 1.0 / diag_totals[d]

        if self._props["model"] == "org":
            self._model = gensim.models.Word2Vec(sentences, sg=skipgram, window=self._window,
                                                 iter=5, size=self._size, min_count=1, workers=20)
        else:
            pre = filename + "_pre"
            suf = filename + "_suf"
            self._model = Word2Vec(sentences, sg=skipgram, window=self._window, iter=5,
                                   size=self._size, min_count=1, workers=20, pre=pre, suf=suf)

    def cross_validate(self, train_files, valid_files):
        self._reset_stats()
        for i, train in enumerate(train_files):
            self.train(train_files[i])
            self.valid(valid_files[i])

    def test(self, test_files):
        for test in test_files:
            print(test)
            with open(test) as f:
                for line in f:
                    feed_events = line.split("|")[2].split(" ")
                    # feed_events = [w for w in feed_events if w not in self._stopwordslist]
                    actual = line.split("|")[0].split(",")
                    predictions = self.predict(feed_events)
                    for diag in self._diags:
                        prior = (diag in feed_events) if self._prior_pred else None

                        if prior is not None:
                            predictions[diag] *= abs((self._prior[diag] - int(not prior)))

                        self._true_test[diag].append((diag in actual))
                        prediction = (predictions[diag] - self._mean[diag]) / self._std[diag]
                        self._pred_test[diag].append(prediction)
        self._store_tmp()

    def valid(self, filename):
        with open(filename) as f:
            for line in f:
                feed_events = line.split("|")[2].split(" ")
                # feed_events = [w for w in feed_events if w not in self._stopwordslist]
                actual = line.split("|")[0].split(",")
                predictions = self.predict(feed_events)
                for diag in self._diags:
                    prior = (diag in feed_events) if self._prior_pred else None
                    self.stat_prediction(predictions[diag], (diag in actual), diag, prior)

    def stat_prediction(self, prediction, actual, diag, prior=None):
        if prior is not None:
            prediction *= abs((self._prior[diag] - int(not prior)))

        self._true_vals[diag].append(actual)
        self._pred_vals[diag].append(prediction)

    def _remove_stopwords(self, sentences):
        '''
        This function has shown over and over again that it is not useful
        '''
        self._word_counter = defaultdict(lambda: 0)
        for sentence in sentences:
            for word in sentence:
                self._word_counter[word] += 1

        inverse = {v: k for k, v in self._word_counter.items()}
        topwords = sorted(inverse.keys(), reverse=True)[:self._stopwords]
        self._stopwordslist = [inverse[k] for k in topwords]

        newsentences = []
        for s in sentences:
            newsentences.append([w for w in s if w not in self._stopwordslist])

        return newsentences

    @property
    def prediction_per_patient(self):
        return (1.0 * self._total_predictions / (self._miss + self._hit))

    @property
    def accuracy(self):
        return (1.0 * self._hit / (self._miss + self._hit))

    @property
    def name(self):
        fname = self.__class__.__name__
        for k in sorted(self._props):
            fname += "_" + k[:2] + "=" + str(self._props[k])
        return fname

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def _normalize(self, d):
        # self._pred_vals[d] = stats.zscore(self._pred_vals[d])
        # self._mean[d] = mean(self._pred_vals[d])
        # self._std[d] = std(self._pred_vals[d])
        # self._pred_vals[d] = list(map(self.sigmoid, self._pred_vals[d]))
        precision, recall, thresholds = metrics.precision_recall_curve(
            self._true_vals[d], self._pred_vals[d])
        max_f1_score = 0
        for i in range(len(precision)):
            if precision[i] == 0 or recall[i] == 0:
                f1_score = 0
            else:
                f1_score = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

            if f1_score > max_f1_score:
                max_f1_score = f1_score
                self._diag_thresholds[d] = thresholds[i]
                self._diag_f1_scores[d] = f1_score

    def _calculate_stats(self):
        self._diag_thresholds = {}
        self._diag_f1_scores = {}
        self._mean = {}
        self._std = {}

        for d in self._diags:
            self._normalize(d)

            for i in range(len(self._true_vals[d])):
                prob = bool(self._pred_vals[d][i] >= self._diag_thresholds[d])
                true_condition = self._true_vals[d][i]

                if prob:
                    self._total_predictions += 1

                if prob is True and true_condition is True:
                    self._stats[d]["TP"] += 1
                    self._hit += 1
                elif prob is False and true_condition is True:
                    self._miss += 1
                    self._stats[d]["FN"] += 1
                elif prob is True and true_condition is False:
                    self._miss += 1
                    self._stats[d]["FP"] += 1
                elif prob is False and true_condition is False:
                    self._hit += 1
                    self._stats[d]["TN"] += 1
                else:
                    assert False, "This shouldnt happen"

    def _store_tmp(self):
        output = open('tmp/' + self.name, 'wb')
        pickle.dump(self._pred_vals, output)
        pickle.dump(self._true_vals, output)
        pickle.dump(self._pred_test, output)
        pickle.dump(self._true_test, output)
        output.close()

    def plot_roc(self):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for d in self._diags:
            fpr[d], tpr[d], _ = metrics.roc_curve(self._true_vals[d], self._pred_vals[d])
            roc_auc[d] = metrics.auc(fpr[d], tpr[d])

        plt.figure(figsize=(12, 12), dpi=120)
        for d in ["d_250", "d_272", "d_311", "d_285", "d_427", "d_428", "d_564"]:
            plt.plot(fpr[d], tpr[d], label='{0} (area = {1:0.3f})'
                     .format(self._diag_to_desc[d], roc_auc[d]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right", fontsize=12)
        plt.savefig('../Results/Plots/ROC_' + self.name + '.png')

    def _report_accuracy(self):
        with open('../Results/accuracies.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            props = {k: self._props[k] for k in self._props}
            props["model"] = self.__class__.__name__
            row = {}
            for d in self._diags:
                row[d] = self._d_auc(d)

            sorted_row = sorted(row.items(), key=operator.itemgetter(1), reverse=True)
            top_auc = [i[1] for i in sorted_row[:10]]
            top_diag = [i[0] for i in sorted_row[:10]]
            writer.writerow([self.accuracy, json.dumps(props, sort_keys=True),
                             self.prediction_per_patient, top_auc, top_diag])

    def write_stats(self):
        self._calculate_stats()
        self._report_accuracy()
        with open('../Results/Stats/' + self.name + '.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            header = ["Diagnosis", "Description", "AUC", "F-Score", "Threshold", "Specificity",
                      "Sensitivity", "Accuracy", "True Positives", "True Negatives",
                      "False Positives", "False Negatives"]
            writer.writerow(header)
            for d in sorted(self._diags):
                row = []
                row.append(d)
                row.append(self._diag_to_desc[d])
                row.append(self._d_auc(d))
                row.append(self._d_fscore(d))
                row.append(self._diag_thresholds[d])
                row.append(self._d_specificity(d))
                row.append(self._d_sensitivity(d))
                row.append(self._d_accuracy(d))
                row.append(self._stats[d]["TP"])
                row.append(self._stats[d]["TN"])
                row.append(self._stats[d]["FP"])
                row.append(self._stats[d]["FN"])
                writer.writerow(row)

    def _d_auc(self, d):
        return (metrics.roc_auc_score(self._true_vals[d], self._pred_vals[d]))

    def _d_sensitivity(self, d):
        if self._stats[d]["TP"] + self._stats[d]["FN"] == 0:
            return (self._stats[d]["TP"] / 1.0)
        else:
            return (self._stats[d]["TP"]*1.0 / (self._stats[d]["TP"] + self._stats[d]["FN"]))

    def _d_specificity(self, d):
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
