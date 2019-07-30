# Fully based on http://www.erogol.com/predict-probabilities-sklearn-linearsvc/. (Accessed on July 29, 2019.) (Posted on November 14, 2014.)

import numpy as np
from sklearn.svm import LinearSVC

class LinearSVC_proba(LinearSVC):

    def __platt_func(self,x):
        return 1/(1+np.exp(-x))

    def predict_proba(self, X):
        f = np.vectorize(self.__platt_func)
        raw_predictions = self.decision_function(X)
        platt_predictions = f(raw_predictions)
        probs = platt_predictions / platt_predictions.sum(axis=1)[:, None]
        return probs
