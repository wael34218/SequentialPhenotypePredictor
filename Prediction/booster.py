from binarypredictor import BinaryPredictor
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import VotingClassifier
from sklearn import linear_model
import numpy as np


class Booster(BinaryPredictor):
    def __init__(self, filename):
        self._props = {"balanced": True}
        self._window = 1
        self._size = 10
        super(Booster, self).__init__(filename)
        self.base_train(filename)

    def train(self, filename):
        print("Training")
        cbowsim = open('tmp/CbowSim', 'rb')
        cbowsim_pred_valid = pickle.load(cbowsim)
        self._valid_labels = pickle.load(cbowsim)
        cbowsim_pred_test = pickle.load(cbowsim)
        self._test_labels = pickle.load(cbowsim)
        cbowsim.close()

        skipgram = open('tmp/SkipGram', 'rb')
        skipgram_pred_valid = pickle.load(skipgram)
        self._valid_labels = pickle.load(skipgram)
        skipgram_pred_test = pickle.load(skipgram)
        skipgram.close()

        collaborative = open('tmp/Collaborative', 'rb')
        collaborative_pred_valid = pickle.load(collaborative)
        self._valid_labels = pickle.load(collaborative)
        collaborative_pred_test = pickle.load(collaborative)
        collaborative.close()

        self.X_test = {}
        self.y_test = {}
        self.models = {}
        for d in self._diags:
            y_train = np.array([int(x) for x in self._valid_labels[d]])
            X_train = np.array([[x, y, z] for x, y, z in zip(cbowsim_pred_valid[d],
                                                             skipgram_pred_valid[d],
                                                             collaborative_pred_valid[d])])

            # params = {'n_estimators': 500, 'max_depth': 14, 'min_samples_split': 5,
            #                     'learning_rate': 0.1, 'loss': 'ls'}
            # self.models[d] = GradientBoostingRegressor(**params)
            # self.models[d] = linear_model.LogisticRegression()
            # self.models[d] = VotingClassifier(voting='soft')
            self.models[d] = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm='SAMME.R', n_estimators=200)
            self.models[d].fit(X_train, y_train)

            self.y_test[d] = self._test_labels[d]
            self.X_test[d] = np.array([[x, y, z] for x, y, z in zip(cbowsim_pred_test[d],
                                                                    skipgram_pred_test[d],
                                                                    collaborative_pred_test[d])])

    def valid(self, filename):
        print("testing")
        for diag in self._diags:
            self._true_vals[diag] = self.y_test[diag]
            self._pred_vals[diag] = [x[1] for x in self.models[diag].predict_proba(self.X_test[diag]).tolist()]


if __name__ == '__main__':
    data_path = "../Data/seq_combined_balanced/"
    model = Booster(data_path + 'mimic_train_0')

    model.cross_validate(["1"], ["1"])
    model.write_stats()
    print(model.accuracy)
    model.plot_roc()
