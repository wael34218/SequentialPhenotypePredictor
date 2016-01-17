import gensim
import argparse
import math
from predictor import Predictor


class CbowSim(Predictor):

    def __init__(self, filename, window=600, size=600, decay=5, stopwords=5):
        self._window = window
        self._size = size
        self._decay = decay
        self._stopwords = stopwords
        self._props = {"window": window, "size": size, "decay": decay, "stopwords": stopwords}
        super(CbowSim, self).__init__(filename)

    def train(self, filename):
        self._sim_mat = {}
        self._filename = filename

        from collections import defaultdict
        self._word_counter = defaultdict(lambda: 0)

        with open(filename) as f:
            sentences = [s[:-1].replace(",", "").split(' ') for s in f.readlines()]
            for sentence in sentences:
                for word in sentence:
                    self._word_counter[word] += 1

            inverse = {v: k for k, v in self._word_counter.items()}
            topwords = sorted(inverse.keys(), reverse=True)[:self._stopwords]
            self._stopwordslist = [inverse[k] for k in topwords]

            newsentences = []
            for s in sentences:
                newsentences.append([w for w in s if w not in self._stopwordslist])

            self._model = gensim.models.Word2Vec(newsentences, sg=0, window=self._window,
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

                feed_events = [w for w in feed_events if w not in self._stopwordslist]

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
    args = parser.parse_args()

    train_files = []
    test_files = []
    model = CbowSim('../Data/mimic_train_me_0', args.window, args.size, args.decay, args.stopwords)

    for i in range(10):
        train_files.append('../Data/mimic_train_me_'+str(i))
        test_files.append('../Data/mimic_test_me_'+str(i))

    model.cross_validate(train_files, test_files)
    model.report_accuracy()
    model.write_stats()
