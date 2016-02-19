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
                for e in events:
                    vec = [x + y for x, y in zip(self._model[e].tolist(), vec)]
                self._pat_vec.append([e * 1.0 / len(events) for e in vec])

                for e in line.split("|")[3].replace("\n", "").split(" "):
                    if e.startswith("d_"):
                        self._pat_diag[i][e] = 1

    def predict(self, feed_events):
        predictions = {}
        vec = [0] * self._size
        te = len(feed_events) * 1.0
        for i, e in enumerate(feed_events):
            dec = math.exp(self._decay*(i-te+1)/te)
            vec = [(x + y) * dec / te for x, y in zip(self._model[e].tolist(), vec)]

        sim = []
        for e in self._pat_vec:
            sim.append(sum([(x*y) for x, y in zip(e, vec)]))

        avg_sim = sum(sim)/len(sim)
        for diag in self._diags:
            probability = 0
            norm = 0
            for i, pat_sim in enumerate(sim):
                if diag in self._pat_diag[i]:
                    probability += max(0, pat_sim - avg_sim)
                norm += max(0, pat_sim - avg_sim)

            predictions[diag] = probability * 1.0 / norm
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
