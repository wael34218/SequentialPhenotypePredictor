from collections import defaultdict
import argparse
from binarypredictor import BinaryPredictor


class Prior(BinaryPredictor):
    def __init__(self, filename, balanced=False):
        self._prior_pred = True
        self._balanced = balanced
        self._stopwordslist = []
        self._props = {"balanced": balanced}
        super(Prior, self).__init__(filename)

    def train(self, filename):
        print(filename)
        self._filename = filename
        self._prior = {}
        self._model = None

        diag_totals = defaultdict(lambda: 0)
        diag_joined = defaultdict(lambda: 0)
        sentences = []
        self.seq_count = 0

        with open(filename) as f:
            for s in f:
                self.seq_count += 1
                sentences.append(s.split("|")[2].split(" ") +
                                 s.split("|")[3].replace("\n", "").split(" "))
                next_diags = s.split("|")[0].split(",")
                prev_diags = [e for e in s.split("|")[2].split(" ") if e.startswith("d_")]
                for d in prev_diags:
                    diag_totals[d] += 1
                    if d in next_diags:
                        diag_joined[d] += 1

        for d in diag_totals:
            self._prior[d] = diag_joined[d] * 1.0 / diag_totals[d]

    def predict(self, feed_events):
        predictions = defaultdict(lambda: 1)
        return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prior')
    parser.add_argument('-b', '--balanced', action="store", default=0, type=int,
                        help='Whether to use balanced or not blanaced datasets (0 or 1) default 0')
    args = parser.parse_args()

    data_path = "../Data/ucsd/"
    if args.balanced:
        data_path = "../Data/ucsd_balanced/"

    bal = False if args.balanced == 0 else True
    model = Prior(data_path + 'uniq', bal)

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
