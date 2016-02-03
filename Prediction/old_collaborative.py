import math
import argparse
import gensim
from predictor import Predictor


class CollaborativeFiltering(Predictor):

    def __init__(self, filename, window, size, stopwords):
        self._filename = filename
        self._window = window
        self._size = size
        self._stopwords = stopwords
        self._props = {"window": window, "size": size, "stopwords": stopwords}
        super(CollaborativeFiltering, self).__init__(filename)

    def train(self, filename):
        self._filename = filename
        self._pat_vec = []
        self._pat_avg = []

        with open(filename) as f:
            sentences = [s[:-1].replace(",", "").split(' ') for s in f.readlines()]
            self._model = gensim.models.Word2Vec(sentences, sg=0, window=self._window,
                                                 size=self._size, min_count=1, workers=20)

        self._pat_diag = [{} for _ in range(len(sentences))]
        for i, s in enumerate(sentences):
            vec = [0] * self._size
            d_count = 0
            for e in s:
                vec = [x + y for x, y in zip(self._model[e].tolist(), vec)]
                if e.startswith("d_"):
                    d_count += 1
                    self._pat_diag[i][e] = 1

            self._pat_vec.append([e / len(s) for e in vec])
            self._pat_avg.append(d_count * 1.0 / len(s))

    def predict(self, feed_events):
        vec = [0] * self._size
        d_count = 0
        for e in feed_events:
            vec = [x + y for x, y in zip(self._model[e].tolist(), vec)]
            if e.startswith("d_"):
                d_count += 1

        vec = [e / len(feed_events) for e in vec]
        avg = d_count * 1.0 / len(feed_events)

        sim = []
        for i, e in enumerate(self._pat_vec):
            sim.append(math.sqrt(sum([(x-y)**2 for x, y in zip(e, vec)])))

        prediction = set()
        avg_sim = sum(sim)/len(sim)
        for diag in self._diags:
            probability = 0
            norm = 0
            for i, pat_sim in enumerate(sim):
                if diag in self._pat_diag[i]:
                    probability += max(0, pat_sim - avg_sim)
                norm += max(0, pat_sim - avg_sim)

            if 0.25 < probability * 1.0 / norm:
                prediction.add(diag)

        return prediction

    def test(self, filename):
        with open(filename) as f:
            for line in f:
                feed_index = line[0:line.rfind(" d_")].rfind(",")
                feed_events = line[0:feed_index].replace(",", "").split(" ")
                last_admission = line[feed_index:].replace("\n", "").replace(",", "").split(" ")
                actual = set([x for x in last_admission if x.startswith('d_')])

                prediction = self.predict(feed_events)

                print(prediction)
                print(actual)
                print("==========")
                self.stat_prediction(prediction, actual)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collaborative Filtering')
    parser.add_argument('-w', '--window', action="store", default=10, type=int,
                        help='Set max skip length between words (default: 10)')
    parser.add_argument('-s', '--size', action="store", default=200, type=int,
                        help='Set size of word vectors (default: 200)')
    parser.add_argument('-sw', '--stopwords', action="store", default=5, type=int,
                        help='Set number of stop words (default: 5)')

    args = parser.parse_args()
    model = CollaborativeFiltering('../Data/w2v/mimic_train_me_0',
                                   args.window, args.size, args.stopwords)
    train_files = []
    test_files = []
    for i in range(1):
        train_files.append('../Data/w2v/mimic_train_me_'+str(i))
        test_files.append('../Data/w2v/mimic_test_me_'+str(i))

    model.cross_validate(train_files, test_files)
    model.report_accuracy()
    model.write_stats()
    print(model.accuracy)
    print(model.prediction_per_patient)
