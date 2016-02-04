import gensim
from collections import defaultdict
import argparse
from binarypredictor import BinaryPredictor


class SkipGram(BinaryPredictor):
    def __init__(self, filename, window=100, size=200):
        self._window = window
        self._size = size
        self._props = {"window": window, "size": size}
        super(SkipGram, self).__init__(filename)
        self._threshold = 0.25

    def train(self, filename):
        with open(filename) as f:
            sentences = [s.split('|')[2].split(" ") + s[:-1].split("|")[3].split(" ")
                         for s in f.readlines()]
            self._model = gensim.models.Word2Vec(sentences, sg=1, size=self._size,
                                                 window=self._window, min_count=1, workers=20)

    def test(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                feed_events = line.split("|")[2].split(" ")
                actual = line.split("|")[0].split(",")
                result = defaultdict(
                    lambda: 1, {d: sim for d, sim in self._model.most_similar(
                        feed_events, topn=self._nevents)})

                for diag in self._diags:
                    self.stat_prediction(result[diag], (diag in actual), diag)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SkipGram Similarity')
    parser.add_argument('-w', '--window', action="store", default=10, type=int,
                        help='Set max skip length between words (default: 10)')
    parser.add_argument('-s', '--size', action="store", default=600, type=int,
                        help='Set size of word vectors (default: 600)')
    args = parser.parse_args()

    train_files = []
    test_files = []
    model = SkipGram('../Data/seq_combined/mimic_train_0', args.window, args.size)

    for i in range(10):
        train_files.append('../Data/seq_combined/mimic_train_'+str(i))
        test_files.append('../Data/seq_combined/mimic_test_'+str(i))

    model.cross_validate(train_files, test_files)
    model.report_accuracy()
    model.write_stats()
    print(model.accuracy)
