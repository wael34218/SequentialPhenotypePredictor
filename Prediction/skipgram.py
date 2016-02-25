from collections import defaultdict
import argparse
from binarypredictor import BinaryPredictor


class SkipGram(BinaryPredictor):
    def __init__(self, filename, window=10, size=600, decay=5, balanced=False, prior=True):
        self._window = window
        self._size = size
        self._decay = decay
        self._prior_pred = prior
        self._stopwordslist = []
        self._props = {"window": window, "size": size, "decay": decay, "prior": prior,
                       "balanced": balanced}
        super(SkipGram, self).__init__(filename)

    def train(self, filename):
        self.base_train(filename, skipgram=1)

    def predict(self, feed_events):
        predictions = defaultdict(
            lambda: 1, {d: sim * (sim > 0) for d, sim in self._model.most_similar(
                feed_events, topn=self._nevents)})
        return predictions

#    def word_vector_graph(self, filename):
#        from matplotlib import pyplot as plt
#        fig = plt.figure(figsize=(14, 14), dpi=180)
#        plt.plot()
#        ax = fig.add_subplot(111)
#        for e in self._uniq_events:
#            v = self._model[e]
#            plt.plot(v[0], v[1])
#            ax.annotate(e, xy=v, fontsize=10)
#        plt.savefig('../Results/Plots/event_rep.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SkipGram Similarity')
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

    train_files = []
    test_files = []

    data_path = "../Data/seq_combined/"
    if args.balanced:
        data_path = "../Data/seq_combined_balanced/"

    prior = False if args.prior == 0 else True
    bal = False if args.balanced == 0 else True
    model = SkipGram(data_path + 'mimic_train_0', args.window, args.size, args.decay, bal, prior)

    for i in range(10):
        train_files.append(data_path + 'mimic_train_'+str(i))
        test_files.append(data_path + 'mimic_test_'+str(i))

    model.cross_validate(train_files, test_files)
    model.write_stats()
    print(model.accuracy)
    model.plot_roc()
