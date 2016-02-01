import math
from collections import defaultdict
import argparse
from binarypredictor import BinaryPredictor
from nltk import ngrams
import itertools


class TFIDF(BinaryPredictor):

    def __init__(self, filename, ngrams=3, skip=3, decay=0, balanced=False):
        self._ngrams = ngrams
        self._skip = skip
        self._decay = decay
        self._threshold = 22
        self._props = {"ngrams": ngrams, "decay": decay, "skip": skip}

        super(TFIDF, self).__init__(filename)

    def _generate_grams(self, sequence):
        termc = defaultdict(lambda: 0)
        for ngram in ngrams(sequence, self._ngrams + self._skip, pad_right=True):
            head = ngram[:1]
            tail = ngram[1:]
            for skip_tail in itertools.combinations(tail, self._ngrams - 1):
                if skip_tail[-1] is None:
                    continue
                termc[head + skip_tail] += 1
        return [("".join(t), termc[t]) for t in termc]

    def train(self, filename):
        print("training", filename)
        self._ldiagp = {}
        self._ldiagtermp = {}
        self._lidf = defaultdict(lambda: 0)
        self._termc = defaultdict(lambda: 0)
        diagtermc = {d: defaultdict(lambda: 1) for d in self._diags}
        diagc = defaultdict(lambda: 0)
        total_count = 0

        with open(filename) as f:
            for s in f.readlines():
                total_count += 1
                events = s.split("|")[2].split(" ")
                diags = s.split("|")[0].split(",")
                terms = self._generate_grams(events)
                for t, c in terms:
                    self._termc[t] += c

                for d in diags:
                    diagc[d] += 1
                    for t, c in terms:
                        diagtermc[d][t] += 1

        for d in self._diags:
            termsc = sum(diagtermc[d].values())
            self._ldiagtermp[d] = defaultdict(
                lambda: 0, {t: math.log(v * 1.0 / termsc) for t, v in diagtermc[d].items()})
            self._ldiagp[d] = math.log(diagc[d] * 1.0 / total_count)

        for t in self._termc:
            self._lidf[t] = math.log(total_count / self._termc[t] * 1.0)

    def test(self, filename):
        with open(filename) as f:
            for s in f.readlines():
                events = s.split("|")[2].split(" ")
                diags = s.split("|")[0].split(",")
                terms = self._generate_grams(events)
                for d in self._diags:
                    score = self._ldiagp[d]
                    for t, c in terms:
                        score += math.log(1+c) * self._lidf[t] * self._ldiagtermp[d][t]
                    score = abs(score / 1000.0)

                    actual = int(d in diags)
                    prediction = score
                    self.stat_prediction(prediction, actual, d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TFIDF')
    parser.add_argument('-n', '--ngrams', action="store", default=3, type=int,
                        help='N gram (default: 3)')
    parser.add_argument('-s', '--skip', action="store", default=3, type=int,
                        help='Skipgram (default: 3)')
    parser.add_argument('-d', '--decay', action="store", default=0, type=int,
                        help='decay (default: 0)')
    args = parser.parse_args()

    train_files = []
    test_files = []
    model = TFIDF('../Data/seq_combined/mimic_train_0', args.ngrams, args.skip, args.decay)

    for i in range(10):
        train_files.append('../Data/seq_combined/mimic_train_'+str(i))
        test_files.append('../Data/seq_combined/mimic_test_'+str(i))

    model.cross_validate_combined(train_files, test_files)
    model.report_accuracy()
    print(model.accuracy)
    model.write_stats()
