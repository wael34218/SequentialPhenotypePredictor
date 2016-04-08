from collections import defaultdict
import argparse
from binarypredictor import BinaryPredictor


class Prior(BinaryPredictor):
    def __init__(self, filename, balanced=False, dataset="ucsd"):
        self._prior_pred = True
        self._balanced = balanced
        self._dataset = dataset
        self._stopwordslist = []
        self._props = {"balanced": balanced, "dataset": dataset}
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
        diags = [x for x in feed_events if x.startswith("d_")]
        predictions = defaultdict(lambda: 0)
        for d in diags:
            predictions[d] = 1
        return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prior')
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

    bal = False if args.balanced == 0 else True
    model = Prior(data_path + 'vocab', bal, ds)

    train_files = []
    valid_files = []
    test_files = []
    for i in range(10):
        train_files.append(data_path + 'trainv_'+str(i))
        valid_files.append(data_path + 'test_'+str(i))

    model.cross_validate(train_files, valid_files)
    model.write_stats()
    print(model.accuracy)
