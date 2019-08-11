# Fully based on http://www.erogol.com/predict-probabilities-sklearn-linearsvc/
# (posted on November 14, 2014, and accessed on July 29, 2019),
# except the handling of exception "np.AxisError" which is based on the information given in
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC.decision_function
# (accessed on August 11, 2019).

import numpy as np
from sklearn.svm import LinearSVC

class LinearSVC_proba(LinearSVC):

    def __platt_func(self,x):
        return 1/(1+np.exp(-x))

    def predict_proba(self, X):
        f = np.vectorize(self.__platt_func)
        raw_predictions = self.decision_function(X)
        platt_predictions = f(raw_predictions)
        try:
            probs = platt_predictions / platt_predictions.sum(axis=1)[:, None]
        except np.AxisError:
            probs = np.asarray(list(map(lambda prob1: [1-prob1, prob1], platt_predictions)))
        return probs
