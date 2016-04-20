from collections import defaultdict
import math
import argparse
from binarypredictor import BinaryPredictor
import sys


class SkipGram(BinaryPredictor):
    def __init__(self, filename, window=10, size=600, decay=5, balanced=False, prior=True,
                 dataset="ucsd", model="org"):
        self._window = window
        self._size = size
        self._decay = decay
        self._prior_pred = prior
        self._stopwordslist = []
        self._dataset = dataset
        self._props = {"window": window, "size": size, "decay": decay, "prior": prior,
                       "balanced": balanced, "dataset": dataset, "model": model}
        super(SkipGram, self).__init__(filename)

    def train(self, filename):
        print(filename)
        self.base_train(filename, skipgram=1)

    def predict(self, feed_events):
        te = len(feed_events)
        weighted_events = [(e,  math.exp(self._decay*(i-te+1)/te))
                           for i, e in enumerate(feed_events) if e in self._model.vocab]
        predictions = defaultdict(
            lambda: 1, {d: sim * (sim > 0) for d, sim in self._model.most_similar(
                weighted_events, topn=self._nevents)})
        return predictions

    def word_vector_graph(self, filename):
        from matplotlib import pyplot as plt
        # import seaborn
        self.counts = defaultdict(lambda: 0)
        with open(filename) as f:
            for s in f:
                sentense = s.split("|")[2].split(" ") + s.split("|")[3].replace("\n", "").split(" ")
                for e in sentense:
                    self.counts[e] += 1

        fig = plt.figure(figsize=(14, 14), dpi=180)
        colors = {"d": "black", "p": "blue", "l": "red", "s": "green", "c": "orange"}
        plt.plot()
        ax = fig.add_subplot(111)

        events = []
        for t in ["c", "s", "p", "d", "l"]:
            evs = {e: c for e, c in self.counts.items() if e.startswith(t)}
            events += [x for x, y in sorted(evs.items(), key=lambda k: k[1], reverse=True)[:100]]

        for e in events:
            if e in ["c_V3000", "c_V053", "c_V502", "c_V3001", "c_290", "c_V290"]:
                continue
            if e in self._model.vocab:
                v = self._model[e]
                plt.plot(v[0], v[1])
                ax.annotate(e, xy=v, fontsize=10, color=colors[e[0]])

        p = ax.bar(0, [0], 0, color='blue')
        d = ax.bar(0, [0], 0, color='black')
        c = ax.bar(0, [0], 0, color='orange')
        s = ax.bar(0, [0], 0, color='green')
        l = ax.bar(0, [0], 0, color='red')
        ax.legend((d[0], p[0], l[0], c[0], s[0]), ('Diagnosis', 'Prescription', 'Lab test',
                                                   'Condition', 'Symptom'), loc=4)
        plt.savefig('../Results/Plots/event_'+self._props["dataset"]+'_colored.png')
        sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SkipGram Similarity')
    parser.add_argument('-w', '--window', action="store", default=10, type=int,
                        help='Set max skip length between words (default: 10)')
    parser.add_argument('-s', '--size', action="store", default=600, type=int,
                        help='Set size of word vectors (default: 600)')
    parser.add_argument('-d', '--decay', action="store", default=5, type=float,
                        help='Set exponential decay through time (default: 5)')
    parser.add_argument('-p', '--prior', action="store", default=0, type=int,
                        help='Add prior probability (0 for False, 1 for True) default 0')
    parser.add_argument('-b', '--balanced', action="store", default=0, type=int,
                        help='Whether to use balanced or not blanaced datasets (0 or 1) default 0')
    parser.add_argument('-ds', '--dataset', action="store", default="ucsd", type=str,
                        help='Which dataset to use "ucsd" or "mimic", default "ucsd"')
    parser.add_argument('-m', '--model', action="store", default="org", type=str,
                        help='Which model to use "org" or "chao", default "org"')
    args = parser.parse_args()

    ds = "ucsd"
    if args.dataset == "mimic":
        ds = "mimic"

    data_path = "../Data/" + ds + "_seq/"
    if args.balanced:
        data_path = "../Data/" + ds + "_balanced/"

    prior = False if args.prior == 0 else True
    bal = False if args.balanced == 0 else True
    model = SkipGram(data_path + 'vocab', args.window, args.size, args.decay, bal, prior, ds,
                     args.model)

    train_files = []
    valid_files = []
    test_files = []
    for i in range(10):
        train_files.append(data_path + 'trainv_'+str(i))
        valid_files.append(data_path + 'test_'+str(i))

    model.cross_validate(train_files, valid_files)
    model.write_stats()
    print(model.accuracy)
    model.plot_roc()
