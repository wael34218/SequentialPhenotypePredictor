import argparse
from binarypredictor import BinaryPredictor


class CollaborativeFiltering(BinaryPredictor):
    def __init__(self, filename, window=30, size=200, decay=5, balance=False, prior=True):
        self._window = window
        self._size = size
        self._decay = decay
        self._prior_pred = prior
        self._balance = balance
        self._props = {"window": window, "size": size, "decay": decay, "balanced": balance,
                       "prior": prior}
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
        vec = [0] * self._size
        e_len = len(feed_events) * 1.0
        for e in feed_events:
            vec = [(x + y) / e_len for x, y in zip(self._model[e].tolist(), vec)]

        sim = []
        for e in self._pat_vec:
            sim.append(sum([(x*y) for x, y in zip(e, vec)]))

        avg_sim = sum(sim)/len(sim)

        predictions = {}
        for d in self._diags:
            probability = 0
            norm = 0
            # norm = 0
            for i, pat_sim in enumerate(sim):
                if d in self._pat_diag[i]:
                    probability += max(0, pat_sim - avg_sim)
                norm += max(0, pat_sim - avg_sim)

            predictions[d] = probability * 1.0 / norm

        return predictions

#    def test(self, filename):
#        with open(filename) as f:
#            for line in f:
#                feed_events = line.split("|")[2].split(" ")
#                diags = line.split("|")[0].split(",")
#
#                vec = [0] * self._size
#                e_len = len(feed_events) * 1.0
#                for e in feed_events:
#                    vec = [(x + y) / e_len for x, y in zip(self._model[e].tolist(), vec)]
#
#                sim = []
#                for e in self._pat_vec:
#                    sim.append(sum([(x*y) for x, y in zip(e, vec)]))
#
#                avg_sim = sum(sim)/len(sim)
#                for d in self._diags:
#                    actual = (d in diags)
#                    probability = 0
#                    norm = 0
#                    # norm = 0
#                    for i, pat_sim in enumerate(sim):
#                        if d in self._pat_diag[i]:
#                            probability += max(0, pat_sim - avg_sim)
#                        norm += max(0, pat_sim - avg_sim)
#
#                    prediction = probability * 1.0 / norm
#                    # self.stat_prediction(prediction, actual, d)
#                    self.stat_prediction(prediction, actual, d, (d in feed_events))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collaborative Filtering')
    parser.add_argument('-w', '--window', action="store", default=10, type=int,
                        help='Set max skip length between words (default: 10)')
    parser.add_argument('-s', '--size', action="store", default=600, type=int,
                        help='Set size of word vectors (default: 600)')
    parser.add_argument('-d', '--decay', action="store", default=5, type=float,
                        help='Set exponential decay through time (default: 5)')
    parser.add_argument('-p', '--prior', action="store", default=1, type=int,
                        help='Add prior probability (0 for False, 1 for True) default 1')
    parser.add_argument('-b', '--balanced', action="store", default=0, type=int,
                        help='Whether to use balanced or not blanaced datasets (0 or 1) default 0')
    args = parser.parse_args()

    data_path = "../Data/seq_combined/"
    if args.balanced:
        data_path = "../Data/seq_combined_balanced/"

    train_files = []
    test_files = []

    args = parser.parse_args()
    prior = False if args.prior == 0 else True
    bal = False if args.balanced == 0 else True
    model = CollaborativeFiltering(data_path + 'mimic_train_0', args.window, args.size, args.decay,
                                   bal, prior)

    train_files = []
    valid_files = []
    test_files = []
    for i in range(10):
        train_files.append(data_path + 'mimic_train_'+str(i))
        test_files.append(data_path + 'mimic_test_'+str(i))
        valid_files.append(data_path + 'mimic_valid_'+str(i))

    model.cross_validate(train_files, valid_files)
    model.write_stats()
    print(model.accuracy)
    model.test(test_files)
