import json
import numpy as np
import pytest
from itertools import zip_longest
from os.path import exists
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from tests.utils import generate_available_filename, remove_and_check
from text_categorizer import classifiers

clfs = [
    classifiers.RandomForestClassifier,
    classifiers.BernoulliNB,
    classifiers.MultinomialNB,
    classifiers.ComplementNB,
    classifiers.KNeighborsClassifier,
    classifiers.MLPClassifier,
    classifiers.LinearSVC,
    classifiers.DecisionTreeClassifier,
    classifiers.ExtraTreeClassifier,
    classifiers.DummyClassifier,
    classifiers.SGDClassifier,
    classifiers.BaggingClassifier,
]

def test_classifiers():
    for n_jobs in range(-1, 2):
        for class_weight in [None, 'balanced']:
            for f in clfs:
                clf = f(n_jobs=n_jobs, class_weight=class_weight)
                assert clf.__class__.__name__ == f.__name__ or (clf.__class__.__name__ == 'LinearSVC_proba' and f.__name__  == 'LinearSVC')
                if 'n_jobs' in dir(clf):
                    assert clf.n_jobs == n_jobs
                if 'class_weight' in dir(clf):
                    assert clf.class_weight == class_weight
                if 'random_state' in dir(clf):
                    assert clf.random_state == 42

def test_Pipeline___init__():
    p = classifiers.Pipeline(clfs)
    assert p.classifiers == clfs

def test_Pipeline_start():
    pass

def test_predict_proba_to_dicts():
    X, y = load_digits(n_class=10, return_X_y=True)
    clf = classifiers.LinearSVC(n_jobs=1, class_weight=None)
    with pytest.warns(ConvergenceWarning):
        clf.fit(X, y)
    y_predict_proba = clf.predict_proba(X)
    expected_dicts = [dict(zip_longest(clf.classes_, probs)) for probs in y_predict_proba]
    dicts = classifiers.predict_proba_to_dicts(clf.classes_, y_predict_proba)
    assert all([d1.items() == d2.items() for d1, d2 in zip_longest(dicts, expected_dicts)])

def test_dicts_to_predict():
    n_class = 10
    X, y = load_digits(n_class=n_class, return_X_y=True)
    clf = classifiers.LinearSVC(n_jobs=1, class_weight=None)
    with pytest.warns(ConvergenceWarning):
        clf.fit(X, y)
    y_pred1 = clf.predict(X)
    y_predict_proba = clf.predict_proba(X)
    dicts = classifiers.predict_proba_to_dicts(clf.classes_, y_predict_proba)
    acc_at_1 = accuracy_score(y, y_pred1)
    for n_accepted_probs in range(1, n_class + 1):
        for y_true in [None, y]:
            y_pred2 = None
            try:
                y_pred2 = classifiers.dicts_to_predict(dicts, y_true, n_accepted_probs)
            except AssertionError:
                if y_true is None and 1 < n_accepted_probs <= n_class:
                    continue
                else:
                    raise
            acc = accuracy_score(y, y_pred2)
            if 1 < n_accepted_probs < n_class:
                assert acc > acc_at_1
            elif n_accepted_probs == 1:
                assert np.array_equal(y_pred1, y_pred2)
                assert acc == acc_at_1
            else:
                assert np.array_equal(y, y_pred2)
                assert acc == 1

def test_dump_json():
    d1 = {'test_random_values': [np.random.random()]}
    filename = generate_available_filename()
    try:
        classifiers.dump_json(d1, filename)
        f = open(filename, 'r')
        d2 = json.load(f)
    finally:
        f.close()
        remove_and_check(filename)
    assert d1 == d2

@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_generate_roc_plot():
    filename = '%s.png' % (generate_available_filename())
    for n_class in [2, 10]:
        X_test, y_test = load_digits(n_class=n_class, return_X_y=True)
        for f in clfs:
            clf = f(n_jobs=1, class_weight=None)
            clf.fit(X_test, y_test)
            assert not exists(filename)
            try:
                classifiers.generate_roc_plot(clf, X_test, y_test, filename)
                assert exists(filename)
            finally:
                remove_and_check(filename)
