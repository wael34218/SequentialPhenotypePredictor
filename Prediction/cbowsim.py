import gensim
import argparse
import math
from binarypredictor import BinaryPredictor
from collections import defaultdict


class CbowSim(BinaryPredictor):

    def __init__(self, filename, window=600, size=600, decay=5, stopwords=5,
                 threshold=0.5, balanced=False):
        self._window = window
        self._size = size
        self._decay = decay
        self._threshold = threshold
        self._stopwords = stopwords
        self._stopwordslist = []

        self._sim_mat = {}

        self._balanced = balanced
        self._props = {"window": window, "size": size, "decay": decay, "stopwords": stopwords,
                       "balanced": balanced, "threshold":threshold}
        super(CbowSim, self).__init__(filename)

    def train(self, filename, diagnosis, validation_set):
        if validation_set in self._sim_mat:# and not self._balanced:
            return

        self._filename = filename
        self._word_counter = defaultdict(lambda: 0)

        with open(filename) as f:
            sentences = [s.split("|")[2].split(" ") + s.split("|")[3].replace("\n", "").split(" ")
                         for s in f.readlines()]
            for sentence in sentences:
                for word in sentence:
                    self._word_counter[word] += 1

            newsentences = []
            if self._stopwords == 0:
                newsentences = sentences
            else:
                inverse = {v: k for k, v in self._word_counter.items()}
                topwords = sorted(inverse.keys(), reverse=True)[:self._stopwords]
                self._stopwordslist = [inverse[k] for k in topwords]

                for s in sentences:
                    newsentences.append([w for w in s if w not in self._stopwordslist])

            self._model = gensim.models.Word2Vec(newsentences, sg=0, window=self._window,
                                                 size=self._size, min_count=1, workers=20)

        self._sim_mat[validation_set] = {}
        for diag in self._selected_diags:
            words = self._model.most_similar(diag, topn=len(self._uniq_events))
            sim_array = [0] * len(self._uniq_events)
            sim_array[self._events_index.index(diag)] = 1
            for event, distance in words:
                sim_array[self._events_index.index(event)] = distance
            self._sim_mat[validation_set][diag] = sim_array

    def test(self, filename, diagnosis, validation_set):
        with open(filename) as f:
            for line in f:
                feed_events = line.split("|")[2].split(" ")
                actual = int(line[0])
                test_array = [0] * len(self._uniq_events)

                feed_events = [w for w in feed_events if w not in self._stopwordslist]

                te = len(feed_events)
                for i, e in enumerate(feed_events):
                    test_array[self._events_index.index(e)] += math.exp(self._decay*(i-te+1)/te)

                sim_array = self._sim_mat[validation_set][diagnosis]
                dot_product = sum([(x*y) for x, y in zip(test_array, sim_array)])
                prediction = (dot_product * 100) / self._size
                if prediction > 1:
                    prediction = 1
                elif prediction <0:
                    prediction = 0

                self.stat_prediction(prediction, actual, diagnosis)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CBOW Similarity')
    parser.add_argument('-w', '--window', action="store", default=10, type=int,
                        help='Set max skip length between words (default: 10)')
    parser.add_argument('-s', '--size', action="store", default=600, type=int,
                        help='Set size of word vectors (default: 600)')
    parser.add_argument('-d', '--decay', action="store", default=5, type=float,
                        help='Set exponential decay through time (default: 5)')
    parser.add_argument('-sw', '--stopwords', action="store", default=5, type=int,
                        help='Set number of stop words (default: 5)')
    parser.add_argument('-b', '--balanced', action="store", default=False, type=bool,
                        help='Choose if data set is balanced or not (default: False)')
    parser.add_argument('-t', '--threshold', action="store", default=0.5, type=float,
                        help='Threshold for prediction probability (default: 0.5)')
    args = parser.parse_args()

    train_files = []
    test_files = []
    model = CbowSim('../Data/w2v_diag/mimic_train_d_038_0',
                    args.window, args.size, args.decay, args.stopwords,
                    args.threshold, args.balanced)

    folder = "../Data/w2v_diag_balanced/" if args.balanced else "../Data/w2v_diag/"
    print(folder)
    diags = []
    validation_set = []
    for d in model._selected_diags:
        for i in range(10):
            train_files.append(folder+'mimic_train_'+d+'_'+str(i))
            test_files.append(folder+'mimic_test_'+d+'_'+str(i))
            diags.append(d)
            validation_set.append(i)

    model.cross_validate(train_files, test_files, diags, validation_set)
    model.report_accuracy()
    print(model.accuracy)
    model.write_stats()
