import math
import argparse
from predictor import Predictor


class NaiveBayes(Predictor):

    def __init__(self, filename, ngrams=10, decay=0):
        self._ngrams = ngrams
        self._decay = decay
        self._props = {"ngrams": ngrams, "decay": decay}
        super(NaiveBayes, self).__init__(filename)

    def train(self, filename):
        self._tcounts = [self._nevents * self._ngrams] * self._nevents
        self._ncounts = [[[1] * self._nevents
                         for x in range(self._nevents)]
                         for y in range(self._ngrams)]
        self._filename = filename
        print("start Training")
        recent = [None] * self._ngrams
        total = 0
        with open(filename) as f:
            for s in f.readlines():
                events = s[:-1].split(' ')
                for e in events:
                    total += 1
                    self._tcounts[self._events_index.index(e)] += 1

                    for i in range(self._ngrams):
                        if recent[i] is None:
                            continue
                        self._ncounts[i][self._events_index.index(recent[i])]\
                                        [self._events_index.index(e)] += 1

                    recent.append(e)
                    recent.pop(0)

        for i in range(self._ngrams):
            for pi in range(self._nevents):
                for ni in range(self._nevents):
                    self._ncounts[i][pi][ni] = math.log(self._ncounts[i][pi][ni] * 1.0 /
                                                        self._tcounts[ni])

        self._lcounts = [math.log(e+1/total*1.0) for e in self._tcounts]
        print("Done Training")

    def _calculate_prediction(self, e, feed_events):
        probability = 0
        for i in range(self._ngrams):
            d_factor = math.exp(self._decay*(i-self._ngrams+1)/self._ngrams)
            probability += d_factor * self._ncounts[i][self._events_index.index(feed_events[i])]\
                                                      [self._events_index.index(e)]

        probability += self._lcounts[self._events_index.index(e)]
        return probability

    def test(self, filename):
        with open(filename) as f:
            for line in f:
                feed_index = line[0:line.rfind(" d_")].rfind(",")
                feed_events = line[0:feed_index].replace(",", "").split(" ")[-self._ngrams:]
                if len(feed_events) < self._ngrams:
                    continue
                last_admission = line[feed_index:].replace("\n", "").replace(",", "").split(" ")
                actual = set([x for x in last_admission if x.startswith('d_')])

                prediction = set()
                predicted_seq = []
                while True:
                    max_pred = 0
                    next_word = None
                    for e in [e for e in self._uniq_events if e not in predicted_seq]:
                        prob = self._calculate_prediction(e, feed_events)
                        if prob > max_pred or next_word is None:
                            max_pred = prob
                            next_word = e

                    if next_word.startswith('d_'):
                        prediction.add(next_word)

                    if len(prediction) == 5:
                        break

                    predicted_seq.append(next_word)
                    feed_events.append(next_word)
                    feed_events.pop(0)

                print("========>", prediction)
                self.stat_prediction(prediction, actual)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Naive Bayes')
    parser.add_argument('-n', '--ngrams', action="store", default=10, type=int,
                        help='N gram (default: 10)')
    parser.add_argument('-d', '--decay', action="store", default=0, type=int,
                        help='decay (default: 0)')
    args = parser.parse_args()

    train_files = []
    test_files = []
    model = NaiveBayes('../Data/mimic_train_0', args.ngrams, args.decay)

    for i in range(1):
        train_files.append('../Data/mimic_train_'+str(i))
        test_files.append('../Data/mimic_test_'+str(i))

    model.cross_validate(train_files, test_files)
    model.report_accuracy()
    print(model.accuracy)
    model.write_stats()
    print(model.prediction_per_patient)
