import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from text_categorizer import classifiers
from text_categorizer.LinearSVC_proba import LinearSVC_proba

def test___platt_func():
    clf = LinearSVC_proba()
    __platt_func = clf.__class__.__dict__['_LinearSVC_proba__platt_func']
    assert __platt_func(clf, 0) == 0.5
    assert round(__platt_func(clf, -100), 6) == round(3.720076e-44, 6)
    assert __platt_func(clf, 100) == 1

def test_predict_proba():
    with pytest.warns(ConvergenceWarning):
        return_X_y = True
        for n_class in [2, 10]:
            X, y = load_digits(n_class=n_class, return_X_y=return_X_y)
            clf = LinearSVC_proba()
            clf.fit(X, y)
            y_pred1 = clf.predict(X)
            y_predict_proba = clf.predict_proba(X)
            dicts = classifiers.predict_proba_to_dicts(clf.classes_, y_predict_proba)
            y_pred2 = classifiers.dicts_to_predict(dicts, y_true=None, n_accepted_probs=1)
            assert np.array_equal(y_pred1, y_pred2)
