from collections import defaultdict
import math
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
        print(filename)
        self.base_train(filename, skipgram=1)

    def predict(self, feed_events):
        te = len(feed_events)
        try:
            weighted_events = [(e,  math.exp(self._decay*(i-te+1)/te))
                            for i, e in enumerate(feed_events) if e in self._model.vocab]
            predictions = defaultdict(
                lambda: 1, {d: sim * (sim > 0) for d, sim in self._model.most_similar(
                    weighted_events, topn=self._nevents)})
        except:
            import pdb; pdb.set_trace()
        return predictions

#    def word_vector_graph(self):
#        from matplotlib import pyplot as plt
#        fig = plt.figure(figsize=(14, 14), dpi=180)
#        colors = {"d": "black", "p": "blue", "l": "red"}
#        plt.plot()
#        ax = fig.add_subplot(111)
#        for e in self._uniq_events:
#            if e in ["d_774", "p_AMPVL", "p_NEOSYRD5W", "p_GENT10I"]:
#                continue
#            v = self._model[e]
#            plt.plot(v[0], v[1])
#            ax.annotate(e, xy=v, fontsize=10, color=colors[e[0]])
#        plt.savefig('../Results/Plots/event_rep_colored.png')

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

    data_path = "../Data/ucsd/"
    if args.balanced:
        data_path = "../Data/ucsd_balanced/"

    prior = False if args.prior == 0 else True
    bal = False if args.balanced == 0 else True
    model = SkipGram(data_path + 'uniq', args.window, args.size, args.decay, bal, prior)

    train_files = []
    valid_files = []
    test_files = []
    for i in range(10):
        train_files.append(data_path + 'mimic_trainv_'+str(i))
        valid_files.append(data_path + 'mimic_test_'+str(i))

    model.cross_validate(train_files, valid_files)
    model.write_stats()
    print(model.accuracy)
    model.plot_roc()
