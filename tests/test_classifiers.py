import numpy as np
import pytest
from itertools import zip_longest
from os.path import exists
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from tests.utils import generate_available_filename, remove_and_check
from text_categorizer import classifiers, pickle_manager

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

class FailClassifier:
    def __init__(self, **kwargs):
        pass

    def fit(self, **kwargs):
        raise Exception()

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

@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_Pipeline_start():
    predict_probas_linux = {
        'RandomForestClassifier': [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        'BernoulliNB': [
            [1.0, 5.9253907982022474e-18, 9.24592247679012e-21],
            [5.086117678607322e-14, 0.9999999417850541, 5.821489476394197e-08],
        ],
        'MultinomialNB': [
            [1.0, 3.987155612430403e-87, 1.9843977254102716e-103],
            [1.1638109881136655e-141, 1.0, 4.902906597402722e-42],
        ],
        'ComplementNB': [
            [1.0, 1.244018908413837e-57, 2.372151728763692e-55],
            [1.2983800585685595e-35, 1.0, 3.836692075297123e-24],
        ],
        'KNeighborsClassifier': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        'MLPClassifier': [
            [0.9999992330465266, 2.108350674827178e-08, 7.458699665987544e-07],
            [6.949799904570786e-10, 0.9999171940556058, 8.280524941418183e-05],
        ],
        'LinearSVC': [
            [0.8995782143576087, 0.02511044323694783, 0.07531134240544347],
            [0.03561932795252063, 0.9407083426933305, 0.023672329354149018],
        ],
        'DecisionTreeClassifier': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        'ExtraTreeClassifier': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        'DummyClassifier': [[0, 0, 1], [1, 0, 0]],
        'SGDClassifier': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        'BaggingClassifier': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    }
    predict_probas_windows = {
        'ComplementNB': [
            [1.0, 1.2440189084141198e-57, 2.3721517287642315e-55],
            [1.2983800585685595e-35, 1.0, 3.836692075297123e-24],
        ],
        'MLPClassifier': [
            [0.9999992330465266, 2.108350674827178e-08, 7.458699665987557e-07],
            [6.949799904570761e-10, 0.9999171940556058, 8.280524941418183e-05],
        ],
        'LinearSVC': [
            [0.8995692949536029, 0.025113499736912265, 0.07531720530948487],
            [0.0356197780956943, 0.9407082394988142, 0.02367198240549154],
        ],
    }
    p = classifiers.Pipeline(clfs)
    clfs_names = [f.__name__ for f in p.classifiers]
    clfs_files = ['%s.pkl' % (clf_name) for clf_name in clfs_names]
    roc_files = ['ROC_%s.png' % (clf_name) for clf_name in clfs_names]
    X, y = load_digits(n_class=3, return_X_y=True)
    X = X.tolist()
    y = y.tolist()
    assert all([not exists(clf_name) for clf_name in clfs_names])
    try:
        predictions = p.start(X, y, X, y, -1, {1, 2, 2, 4})
        for clf_name, clf_file in zip_longest(clfs_names, clfs_files):
            predict_proba = [list(d.values()) for d in predictions[clf_name][0:2]]
            assert np.array_equal(predict_probas_linux[clf_name], predict_proba) \
                    or np.array_equal(predict_probas_windows[clf_name], predict_proba)
            assert exists(clf_file)
            clf = pickle_manager.load(clf_file)
            if 'n_jobs' in dir(clf):
                assert clf.n_jobs == -1
            if 'class_weights' in dir(clf):
                assert clf.class_weights is None
        assert all([not exists(roc_file) for roc_file in roc_files])
        p.start(X, y, X, y, -1, {1, 2, 2, 4}, 'balanced')
        for clf_file in clfs_files:
            clf = pickle_manager.load(clf_file)
            if 'class_weights' in dir(clf):
                assert clf.class_weights == 'balanced'
        p.start(X, y, X, y, -1, {1, 2, 2, 4}, None, True)
        assert all([exists(roc_file) for roc_file in roc_files])
        classifiers.Pipeline([FailClassifier]).start(X, y, X, y)
    finally:
        for clf_file in clfs_files:
            remove_and_check(clf_file)
        for roc_file in roc_files:
            remove_and_check(roc_file)

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
