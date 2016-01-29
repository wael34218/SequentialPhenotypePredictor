from predictor import Predictor


class LatentFactors(Predictor):

    def __init__(self, filename):
        self._props = {"window": 200, "size": 60, "decay": 10, "delta": 10, "rounds": 40,
                       "param_reg": 0.015, "latent_reg": 0.015, "factors": 1000}
        super(LatentFactors, self).__init__(filename)
        self._diags = ["d_584", "d_428", "d_272", "d_403", "d_427"]
        self._auc_enabled = True

    def calculate_statistics(self):
        for diag in self._diags:
            for i in range(10):
                pred_file = open("latentfactors/predictions/pred_"+diag+"_"+str(i))
                actual_file = open('../Data/svd_balanced/mimic_test_'+diag+"_"+str(i))
                for pred in pred_file.readlines():
                    prob = (float(pred.strip()) > 0.75)
                    val = (actual_file.readline()[0] == "1")
                    self._true_vals[diag].append(int(val))
                    self._pred_vals[diag].append(float(pred.strip()))

                    if prob is True and val is True:
                        self._stats[diag]["TP"] += 1
                        self._hit += 1
                    elif prob is False and val is True:
                        self._miss += 1
                        self._stats[diag]["FN"] += 1
                    elif prob is True and val is False:
                        self._miss += 1
                        self._stats[diag]["FP"] += 1
                    elif prob is False and val is False:
                        self._hit += 1
                        self._stats[diag]["TN"] += 1
                    else:
                        assert False, "This shouldnt happen"

                pred_file.close()
                actual_file.close()


if __name__ == '__main__':
    model = LatentFactors('../Data/w2v/mimic_train_me_0')
    model.calculate_statistics()
    model.report_accuracy(calculate_true_negatives=False)
    model.write_stats(calculate_true_negatives=False)
    print(model.accuracy)
