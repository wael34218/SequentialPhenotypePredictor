import math
from collections import defaultdict
import argparse
from binarypredictor import BinaryPredictor
from nltk import ngrams
import itertools


class TFIDF(BinaryPredictor):

    def __init__(self, filename, ngrams=3, skip=3, decay=0, stopwords=0, threshold=0,
                 balanced=False):
        self._ngrams = ngrams
        self._skip = skip
        self._decay = decay
        self._stopwords = stopwords
        self._threshold = threshold
        # Stopwords are not actually calculated - added to comply with the same interface as other
        # predictors
        self._stopwordslist = []
        self._props = {"ngrams": ngrams, "decay": decay, "skip": skip, "stopwords": stopwords,
                       "threshold": threshold, "balanced": balanced}
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
                    termc[head + skip_tail] += 1 * math.exp(-1.0 * self._decay * i / total)
            return [("".join(t), termc[t]) for t in termc]
        else:
            total = len(sequence)
            for i, t in enumerate(reversed(sequence)):
                termc[t] += 1 * math.exp(-1.0 * self._decay * i / total)
            return [(t, termc[t]) for t in termc]

    def train(self, filename):
        print("training", filename)
        self._ldiagp = {}
        self._ldiagtermp = {}
        self._lidf = defaultdict(lambda: 0)
        self._termc = defaultdict(lambda: 0)
        diagtermc = {d: defaultdict(lambda: 1) for d in self._diags}
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
                terms = self._generate_grams(events)

                for t, c in terms:
                    terme[t] += 1

                for d in diags:
                    diagc[d] += 1
                    for t, c in terms:
                        diagtermc[d][t] += c

                # for prior probability calculation
                prev_diags = [e for e in s.split("|")[2].split(" ") if e.startswith("d_")]
                for d in prev_diags:
                    diag_totals[d] += 1
                    if d in diags:
                        diag_joined[d] += 1

        for d in self._diags:
            self._prior[d] = diag_joined[d] * 1.0 / diag_totals[d]
            termsc = sum(diagtermc[d].values())
            self._ldiagtermp[d] = defaultdict(
                lambda: 0, {t: math.log((v * 1.0) / termsc) for t, v in diagtermc[d].items()})
            self._ldiagp[d] = math.log(diagc[d] * 1.0 / total_count)

        for t in terme:
            self._lidf[t] = math.log(total_count * 1.0 / terme[t])

    def predict(self, feed_events):
        predictions = {}
        terms = self._generate_grams(feed_events)

        for diag in self._diags:
            score = self._ldiagp[diag]
            for t, c in terms:
                score += c * self._lidf[t] * self._ldiagtermp[diag][t]
            predictions[diag] = score

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
    parser = argparse.ArgumentParser(description='TFIDF')
    parser.add_argument('-n', '--ngrams', action="store", default=3, type=int,
                        help='N gram (default: 3)')
    parser.add_argument('-s', '--skip', action="store", default=3, type=int,
                        help='Skipgram (default: 3)')
    parser.add_argument('-d', '--decay', action="store", default=0.0, type=float,
                        help='decay (default: 0.0)')
    parser.add_argument('-sw', '--stopwords', action="store", default=0, type=int,
                        help='Set number of stop words (default: 0)')
    parser.add_argument('-t', '--threshold', action="store", default=0.0, type=float,
                        help='Decay (default: 0.0)')
    parser.add_argument('-b', '--balanced', action="store", default=False, type=bool,
                        help='Whether to use balanced or not blanaced datasets')
    args = parser.parse_args()

    train_files = []
    test_files = []

    data_path = "../Data/seq_combined/"
    if args.balanced:
        data_path = "../Data/seq_combined_balanced/"

    model = TFIDF(data_path + 'mimic_train_0', args.window, args.size, args.decay,
                  args.stopwords, args.threshold, args.balanced)

    for i in range(10):
        train_files.append(data_path + 'mimic_train_'+str(i))
        test_files.append(data_path + 'mimic_test_'+str(i))

    model.cross_validate(train_files, test_files)
    model.report_accuracy()
    print(model.accuracy)
    model.write_stats()
