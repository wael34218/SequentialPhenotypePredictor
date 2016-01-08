import gensim
import argparse
import math
from predictor import Predictor


class CbowSim(Predictor):

    def __init__(self, filename, window=600, size=600, decay=5):
        self._window = window
        self._size = size
        self._decay = decay
        self._props = {"window": window, "size": size, "decay": decay}
        super().__init__(filename)

    def train(self, filename):
        self._sim_mat = {}
        self._filename = filename
        with open(filename) as f:
            sentences = [s[:-1].split(' ') for s in f.readlines()]
            self._model = gensim.models.Word2Vec(sentences, sg=0, window=self._window,
                                                 size=self._size, min_count=1, workers=20)

        for diag in self._diags:
            words = self._model.most_similar(diag, topn=len(self._uniq_events))
            sim_array = [0] * len(self._uniq_events)
            sim_array[self._events_index.index(diag)] = 1
            for event, distance in words:
                sim_array[self._events_index.index(event)] = distance
            self._sim_mat[diag] = sim_array

    def test(self, filename):
        with open(filename) as f:
            for line in f:
                feed_index = line[0:line.rfind(" d_")].rfind(",")
                feed_events = line[0:feed_index].replace(",", "").split(" ")
                last_admission = line[feed_index:].replace("\n", "").replace(",", "").split(" ")
                actual = set([x for x in last_admission if x.startswith('d_')])
                test_array = [0] * len(self._uniq_events)

                te = len(feed_events)
                for i, e in enumerate(feed_events):
                    test_array[self._events_index.index(e)] += math.exp(self._decay*(i-te+1)/te)

                result = {}
                for diag in self._sim_mat:
                    result[sum([x*y for x, y in zip(test_array, self._sim_mat[diag])])] = diag

                distances = sorted(result.keys(), reverse=True)[:5]
                prediction = set([result[x] for x in distances])
                # prediction |= set([x for x in feed_events if x.startswith('d_')])
                self.stat_prediction(prediction, actual)

        self.calculate_true_negatives()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CBOW Similarity')
    parser.add_argument('-w', '--window', action="store", default=10, type=int,
                        help='Set max skip length between words (default: 10)')
    parser.add_argument('-s', '--size', action="store", default=600, type=int,
                        help='Set size of word vectors (default: 600)')
    parser.add_argument('-d', '--decay', action="store", default=5, type=float,
                        help='Set exponential decay through time (default: 5)')
    args = parser.parse_args()

    train_files = []
    test_files = []
    model = CbowSim('../Data/mimic_train_0', args.window, args.size, args.decay)

    for i in range(10):
        train_files.append('../Data/mimic_train_'+str(i))
        test_files.append('../Data/mimic_test_'+str(i))

    model.cross_validate(train_files, test_files)
    model.report_accuracy()
    model.write_stats()
