import argparse
from collections import defaultdict
from binarypredictor import BinaryPredictor
from skipgram import SkipGram
from cbowsim import CbowSim
from collaborative import CollaborativeFiltering


class Ensemble(BinaryPredictor):
    def __init__(self, filename, window=10, size=600, decay=8.0, balanced=False, prior=True):
        self._window = window
        self._size = size
        self._decay = decay
        self._prior_pred = prior
        self._stopwordslist = []
        self._props = {"window": window, "size": size, "decay": decay,
                       "prior": prior, "balanced": balanced}
        super(Ensemble, self).__init__(filename)

        self.collaborative = {}
        self.skipgram = {}
        self.cbowsim = {}
        self.collaborative = CollaborativeFiltering(filename, 27, 300, decay, balanced, True)
        self.cbowsim = CbowSim(filename, 45, 275, decay, balanced, True)
        self.skipgram = SkipGram(filename, 23, 350, decay, balanced, False)
        self._models = ["collaborative", "cbowsim", "skipgram"]

    def train(self, filename):
        self.collaborative.train(filename)
        self.cbowsim.train(filename)
        self.skipgram.train(filename)
        self._prior = self.cbowsim._prior
        self._weights = {m: defaultdict(lambda: 0) for m in self._models}

        with open(filename) as f:
            for line in f:
                feed_events = line.split("|")[2].split(" ")
                actual = line.split("|")[0].split(",")

                cf_preds = self.collaborative.predict(feed_events)
                cbow_preds = self.cbowsim.predict(feed_events)
                skip_preds = self.skipgram.predict(feed_events)

                for diag in self._diags:
                    if diag in actual:
                        self._weights["collaborative"][diag] += cf_preds[diag]
                        self._weights["cbowsim"][diag] += cbow_preds[diag]
                        self._weights["skipgram"][diag] += skip_preds[diag]
                    else:
                        self._weights["collaborative"][diag] += 1 - cf_preds[diag]
                        self._weights["cbowsim"][diag] += 1 - cbow_preds[diag]
                        self._weights["skipgram"][diag] += 1 - skip_preds[diag]

            # Normalize weights
            for diag in self._diags:
                norm = (self._weights["collaborative"][diag] + self._weights["cbowsim"][diag] +
                        self._weights["skipgram"][diag])
                self._weights["collaborative"][diag] /= norm
                self._weights["cbowsim"][diag] /= norm
                self._weights["skipgram"][diag] /= norm

            print(self._weights)

    def predict(self, feed_events):
        cf_preds = self.collaborative.predict(feed_events)
        cbow_preds = self.cbowsim.predict(feed_events)
        skip_preds = self.skipgram.predict(feed_events)
        predictions = {}
        for diag in self._diags:
            predictions[diag] = cf_preds[diag] * self._weights["collaborative"][diag]
            predictions[diag] += cbow_preds[diag] * self._weights["cbowsim"][diag]
            predictions[diag] += skip_preds[diag] * self._weights["skipgram"][diag]
        return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble Model')
    parser.add_argument('-w', '--window', action="store", default=10, type=int,
                        help='Set max skip length between words (default: 10)')
    parser.add_argument('-s', '--size', action="store", default=600, type=int,
                        help='Set size of word vectors (default: 600)')
    parser.add_argument('-d', '--decay', action="store", default=8.0, type=float,
                        help='Set exponential decay through time (default: 8.0)')
    parser.add_argument('-p', '--prior', action="store", default=1, type=int,
                        help='Add prior probability (0 for False, 1 for True) default 1')
    parser.add_argument('-b', '--balanced', action="store", default=1, type=int,
                        help='Whether to use balanced or not blanaced datasets (0 or 1) default 1')
    args = parser.parse_args()

    train_files = []
    test_files = []
    data_path = "../Data/seq_combined/"
    if args.balanced:
        data_path = "../Data/seq_combined_balanced/"

    prior = False if args.prior == 0 else True
    bal = False if args.balanced == 0 else True
    model = Ensemble(data_path + '/mimic_train_0', args.window, args.size, args.decay, bal, prior)

    for i in range(10):
        train_files.append(data_path + 'mimic_train_'+str(i))
        test_files.append(data_path + 'mimic_test_'+str(i))

    model.cross_validate(train_files, test_files)
    model.write_stats()
    print(model.accuracy)
    model.plot_roc()
