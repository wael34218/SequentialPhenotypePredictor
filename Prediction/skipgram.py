import gensim
import argparse
from predictor import Predictor


class SkipGram(Predictor):
    def __init__(self, filename, window=100, size=200):
        self._window = window
        self._size = size
        self._props = {"window": window, "size": size}
        super(SkipGram, self).__init__(filename)

    def train(self, filename):
        with open(filename) as f:
            sentences = [s[:-1].split(' ') for s in f.readlines()]
            self._model = gensim.models.Word2Vec(sentences, sg=1, size=self._size,
                                                 window=self._window, min_count=1, workers=20)

    def test(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                feed_index = line[0:line.rfind(" d_")].rfind(",")
                feed_events = line[0:feed_index].replace(",", "").split(" ")

                last_admission = line[feed_index:].replace("\n", "").replace(",", "").split(" ")
                actual = set([x for x in last_admission if x.startswith('d_')])
                result = self._model.most_similar(feed_events, topn=100)
                prediction = set([])
                prediction |= set([x for x, d in result if x.startswith('d_')][:4])
                prediction |= set([x for x in feed_events if x.startswith('d_')])

                self.stat_prediction(prediction, actual)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SkipGram Similarity')
    parser.add_argument('-w', '--window', action="store", default=10, type=int,
                        help='Set max skip length between words (default: 10)')
    parser.add_argument('-s', '--size', action="store", default=600, type=int,
                        help='Set size of word vectors (default: 600)')
    args = parser.parse_args()

    train_files = []
    test_files = []
    model = SkipGram('../Data/mimic_train_0', args.window, args.size)

    for i in range(10):
        train_files.append('../Data/mimic_train_'+str(i))
        test_files.append('../Data/mimic_test_'+str(i))

    model.cross_validate(train_files, test_files)
    model.report_accuracy()
    model.write_stats()
    print(model.accuracy)
