import math
from collections import defaultdict
import argparse
from binarypredictor import BinaryPredictor
from nltk import ngrams
import itertools


class TFIDF(BinaryPredictor):

    def __init__(self, filename, ngrams=3, skip=3, decay=0, balanced=False, prior=True,
                 dataset="ucsd"):
        self._ngrams = ngrams
        self._skip = skip
        self._decay = decay
        self._prior_pred = prior
        self._stopwordslist = []
        self._dataset = dataset
        self._cutoff = 2 * (skip + ngrams)
        self._props = {"ngrams": ngrams, "decay": decay, "skip": skip, "prior": prior,
                       "balanced": balanced, "cutoff": self._cutoff, "dataset": dataset}

        super(TFIDF, self).__init__(filename)

    def _generate_grams(self, sequence):
        termc = defaultdict(lambda: 0)
        if self._ngrams > 1:
            all_seq = list(reversed(list(
                ngrams(sequence, self._ngrams + self._skip, pad_right=True))))
            total = len(all_seq) * 1.0
            for i, ngram in enumerate(all_seq):
                head = ngram[:1]
                tail = ngram[1:]
                for skip_tail in itertools.combinations(tail, self._ngrams - 1):
                    if skip_tail[-1] is None:
                        continue

                    j = total - i + 1
                    termc[head + skip_tail] += 1 * math.exp(-1.0 * self._decay * (total / j))
                    # termc[head + skip_tail] += 1 * math.exp(-1.0 * self._decay *  total / i)
            return [("".join(t), termc[t]) for t in termc]
        else:
            total = len(sequence)
            for i, t in enumerate(reversed(sequence)):
                termc[t] += 1 * math.exp(-1.0 * self._decay * i / total)
            return [(t, termc[t]) for t in termc]

    def train(self, filename):
        print("training", filename)
        self._ldiagp = {}
        self._lndiagp = {}
        self._ldiagtermp = {}
        self._lndiagtermp = {}
        self._lidf = defaultdict(lambda: 0)
        self._termc = defaultdict(lambda: 0)
        diagtermc = {d: defaultdict(lambda: 1) for d in self._diags}
        ndiagtermc = {d: defaultdict(lambda: 1) for d in self._diags}
        diagc = defaultdict(lambda: 0)
        total_count = 0

        terme = defaultdict(lambda: 0)

        self._prior = {}
        diag_joined = defaultdict(lambda: 0)
        diag_totals = defaultdict(lambda: 0)

        with open(filename) as f:
            for s in f.readlines():
                total_count += 1
                events = s.split("|")[2].split(" ")
                diags = s.split("|")[0].split(",")
                terms = self._generate_grams(events[:self._cutoff])

                for d in diags:
                    diagc[d] += 1

                for t, c in terms:
                    terme[t] += 1
                    for d in self._diags:
                        if d in diags:
                            diagtermc[d][t] += c
                        else:
                            ndiagtermc[d][t] += c

                # for prior probability calculation
                prev_diags = [e for e in s.split("|")[2].split(" ") if e.startswith("d_")]
                for d in prev_diags:
                    diag_totals[d] += 1
                    if d in diags:
                        diag_joined[d] += 1

        for d in self._diags:
            self._prior[d] = diag_joined[d] * 1.0 / diag_totals[d]
            termsc = sum(diagtermc[d].values())
            ntermsc = sum(ndiagtermc[d].values())
            self._ldiagtermp[d] = defaultdict(
                lambda: 0, {t: math.log((v * 1.0) / termsc) for t, v in diagtermc[d].items()})
            self._lndiagtermp[d] = defaultdict(
                lambda: 0, {t: math.log((v * 1.0) / ntermsc) for t, v in ndiagtermc[d].items()})
            self._ldiagp[d] = math.log(diagc[d] * 1.0 / total_count)
            self._lndiagp[d] = math.log((total_count - diagc[d] * 1.0) / total_count)

        for t in terme:
            self._lidf[t] = math.log(total_count * 1.0 / terme[t])

    def predict(self, feed_events):
        predictions = {}
        terms = self._generate_grams(feed_events[:self._cutoff])

        for diag in self._diags:
            score = self._ldiagp[diag]
            nscore = self._lndiagp[diag]
            for t, c in terms:
                score += c * self._lidf[t] * self._ldiagtermp[diag][t]
                nscore += c * self._lidf[t] * self._lndiagtermp[diag][t]

            predictions[diag] = abs(nscore) / abs(nscore + score)

        return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TFIDF')
    parser.add_argument('-n', '--ngrams', action="store", default=3, type=int,
                        help='N gram (default: 3)')
    parser.add_argument('-s', '--skip', action="store", default=3, type=int,
                        help='Skipgram (default: 3)')
    parser.add_argument('-d', '--decay', action="store", default=0.0, type=float,
                        help='decay (default: 0.0)')
    parser.add_argument('-p', '--prior', action="store", default=0, type=int,
                        help='Add prior probability (0 for False, 1 for True) default 0')
    parser.add_argument('-b', '--balanced', action="store", default=0, type=int,
                        help='Whether to use balanced or not blanaced datasets (0 or 1) default 0')
    parser.add_argument('-ds', '--dataset', action="store", default="ucsd", type=str,
                        help='Which dataset to use "ucsd" or "mimic", default "ucsd"')
    args = parser.parse_args()

    ds = "ucsd"
    if args.dataset == "mimic":
        ds = "mimic"

    data_path = "../Data/" + ds + "_seq/"
    if args.balanced:
        data_path = "../Data/" + ds + "_balanced/"

    prior = False if args.prior == 0 else True
    bal = False if args.balanced == 0 else True
    model = TFIDF(data_path + 'vocab', args.ngrams, args.skip, args.decay, bal, prior, ds)

    train_files = []
    valid_files = []
    for i in range(10):
        train_files.append(data_path + 'trainv_'+str(i))
        valid_files.append(data_path + 'test_'+str(i))

    model.cross_validate(train_files, valid_files)
    model.write_stats()
    print(model.accuracy)
    model.plot_roc()
