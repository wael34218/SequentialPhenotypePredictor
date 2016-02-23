import argparse
import math
from binarypredictor import BinaryPredictor


class CollaborativeFiltering(BinaryPredictor):
    def __init__(self, filename, window=10, size=600, decay=5, stopwords=0, balanced=False):
        self._window = window
        self._size = size
        self._decay = decay
        self._stopwords = stopwords
        self._stopwordslist = []
        self._props = {"window": window, "size": size, "decay": decay, "stopwords": stopwords,
                       "balanced": balanced}
        super(CollaborativeFiltering, self).__init__(filename)

    def train(self, filename):
        print("Train", filename)
        self.base_train(filename)

        # For patient vectors
        self._pat_diag = [{} for _ in range(self.seq_count)]
        self._pat_vec = []
        with open(filename) as f:
            for i, line in enumerate(f.readlines()):
                vec = [0] * self._size

                events = line.split("|")[2].split(" ")
                te = len(events) * 1.0
                for i, e in enumerate(events):
                    dec = math.exp(self._decay*(i-te+1)/te)
                    vec = [(x + y) * dec / te for x, y in zip(self._model[e].tolist(), vec)]
                self._pat_vec.append([e * 1.0 / len(events) for e in vec])

                for d in line.split("|")[0].split(","):
                    self._pat_diag[i][d] = 1

    def predict(self, feed_events):
        predictions = {}
        vec = [0] * self._size
        te = len(feed_events) * 1.0
        for i, e in enumerate(feed_events):
            dec = math.exp(self._decay*(i-te+1)/te)
            vec = [(x + y) * dec / te for x, y in zip(self._model[e].tolist(), vec)]

        sim = []
        sum_test = math.sqrt(sum([x**2 for x in vec]))
        for e in self._pat_vec:
            dot = sum([(x*y) for x, y in zip(e, vec)])
            sum_pat_vec = math.sqrt(sum([x**2 for x in e]))
            sim.append(dot/(sum_test + sum_pat_vec))

        top_5 = sorted(sim)[5:]
        indexes = [sim.index(i) for i in top_5]

        for diag in self._diags:
            probability = 0
            for i in indexes:
                if diag in self._pat_diag[i]:
                    probability += sim[i]

            predictions[diag] = probability * 1.0 / 5
        return predictions

    def test(self, filename):
        with open(filename) as f:
            for line in f:
                feed_events = line.split("|")[2].split(" ")
                feed_events = [w for w in feed_events if w not in self._stopwordslist]
                actual = line.split("|")[0].split(",")
                predictions = self.predict(feed_events)
                for diag in self._diags:
                    self.stat_prediction(predictions[diag], (diag in actual), diag,
                                         (diag in feed_events))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collaborative Filtering')
    parser.add_argument('-w', '--window', action="store", default=10, type=int,
                        help='Set max skip length between words (default: 10)')
    parser.add_argument('-s', '--size', action="store", default=600, type=int,
                        help='Set size of word vectors (default: 600)')
    parser.add_argument('-d', '--decay', action="store", default=5, type=float,
                        help='Set exponential decay through time (default: 5)')
    parser.add_argument('-sw', '--stopwords', action="store", default=0, type=int,
                        help='Set number of stop words (default: 0)')
    parser.add_argument('-b', '--balanced', action="store", default=False, type=bool,
                        help='Whether to use balanced or not blanaced datasets')
    args = parser.parse_args()

    train_files = []
    test_files = []

    data_path = "../Data/seq_combined/"
    if args.balanced:
        data_path = "../Data/seq_combined_balanced/"

    model = CollaborativeFiltering(data_path + 'mimic_train_0', args.window, args.size, args.decay,
                                   args.stopwords, args.balanced)

    for i in range(10):
        train_files.append(data_path + 'mimic_train_'+str(i))
        test_files.append(data_path + 'mimic_test_'+str(i))

    model.cross_validate(train_files, test_files)
    model.write_stats()
    print(model.accuracy)
