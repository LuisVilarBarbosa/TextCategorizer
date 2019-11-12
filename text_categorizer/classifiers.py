#!/usr/bin/python3
# coding=utf-8

import json
from collections import Counter
from pandas import DataFrame
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from text_categorizer import pickle_manager
from text_categorizer.constants import random_state
from text_categorizer.logger import logger
from text_categorizer.plot_roc import plot_roc
from time import time

def RandomForestClassifier(n_jobs, class_weight):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None,
                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                min_impurity_split=None, bootstrap=True, oob_score=False,
                n_jobs=n_jobs, random_state=random_state, verbose=0,
                warm_start=False, class_weight=class_weight)
    return clf

def BernoulliNB(n_jobs, class_weight):
    from sklearn.naive_bayes import BernoulliNB
    clf = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
    return clf

def MultinomialNB(n_jobs, class_weight):
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    return clf

def ComplementNB(n_jobs, class_weight):
    from sklearn.naive_bayes import ComplementNB
    clf = ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)
    return clf

def KNeighborsClassifier(n_jobs, class_weight):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto',
                leaf_size=30, p=2, metric='minkowski', metric_params=None,
                n_jobs=n_jobs)
    return clf

def MLPClassifier(n_jobs, class_weight):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam',
                alpha=0.0001, batch_size='auto', learning_rate='constant',
                learning_rate_init=0.001, power_t=0.5, max_iter=300, shuffle=True, # Predefined max_iter: 200
                random_state=random_state, tol=0.0001, verbose=False, warm_start=False,
                momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                n_iter_no_change=10)
    return clf

def LinearSVC(n_jobs, class_weight):
    from text_categorizer.LinearSVC_proba import LinearSVC_proba
    clf = LinearSVC_proba(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001,
                C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1,
                class_weight=class_weight, verbose=0, random_state=random_state, max_iter=1000)
    return clf

def DecisionTreeClassifier(n_jobs, class_weight):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None,
                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                max_features=None, random_state=random_state, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None, class_weight=class_weight,
                presort=False)
    return clf

def ExtraTreeClassifier(n_jobs, class_weight):
    from sklearn.tree import ExtraTreeClassifier
    clf = ExtraTreeClassifier(criterion='gini', splitter='random', max_depth=None,
                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                max_features='auto', random_state=random_state, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None, class_weight=class_weight)
    return clf

def DummyClassifier(n_jobs, class_weight):
    from sklearn.dummy import DummyClassifier
    clf = DummyClassifier(strategy='stratified', random_state=random_state, constant=None)
    return clf

def SGDClassifier(n_jobs, class_weight):
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.0001, l1_ratio=0.15,
                fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True,
                verbose=0, epsilon=0.1, n_jobs=n_jobs,
                random_state=random_state, learning_rate='optimal', eta0=0.0, power_t=0.5,
                early_stopping=False, validation_fraction=0.1, n_iter_no_change=5,
                class_weight=class_weight, warm_start=False, average=False, n_iter=None)
    return clf

def BaggingClassifier(n_jobs, class_weight):
    from sklearn.ensemble import BaggingClassifier
    clf = BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0,
                max_features=1.0, bootstrap=True, bootstrap_features=False,
                oob_score=False, warm_start=False, n_jobs=n_jobs,
                random_state=random_state, verbose=0)
    return clf

class Pipeline():
    def __init__(self, classifiers):
        self.classifiers = classifiers
    
    def start(self, X_train, y_train, X_test, y_test, n_jobs=None, set_n_accepted_probs={1,2,3}, class_weight=None, generate_roc_plots=False):
        logger.debug("Number of training examples: %s" % (Counter(y_train)))
        predictions = DataFrame({'y_true': y_test})
        set_n_accepted_probs = set(set_n_accepted_probs)
        new_set_n_accepted_probs = set([elem for elem in set_n_accepted_probs if elem < len(set(y_train).union(y_test))])
        if len(new_set_n_accepted_probs) != len(set_n_accepted_probs):
            set_n_accepted_probs = new_set_n_accepted_probs
            logger.warning("The accepted probabilities set has been changed to not exceed the number of classes.")
        for f in self.classifiers:
            logger.info("Starting %s." % (f.__name__))
            clf = f(n_jobs=n_jobs, class_weight=class_weight)
            logger.debug("%s configuration: %s" % (f.__name__, clf.__dict__))
            t1 = time()
            try:
                clf_filename = "%s.pkl" % (f.__name__)
                predictions_key = "y_pred_%s" % (f.__name__)
                clf.fit(X_train, y_train)
                pickle_manager.dump(clf, clf_filename)
                y_predict_proba = clf.predict_proba(X_test)
                for n_accepted_probs in set_n_accepted_probs:
                    y_predict = predict_proba_to_predict(clf.classes_, y_predict_proba, y_test, n_accepted_probs)
                    if n_accepted_probs == 1:
                        predictions[predictions_key] = y_predict
                        if generate_roc_plots:
                            generate_roc_plot(clf, X_test, y_test, 'ROC_%s.png' % (f.__name__))
                    logger.debug("Confusion matrix:\n%s" % confusion_matrix(y_test, y_predict))
                    logger.debug("Classification report:\n%s" % classification_report(y_test, y_predict))
                    acc = accuracy_score(y_test, y_predict, normalize=True)
                    logger.info("%s: %s | %ss" % (f.__name__, acc, (time() - t1)))
            except Exception as e:
                logger.error("%s: %s | %ss" % (f.__name__, repr(e), (time() - t1)))
        dump_json(predictions, 'predictions.json')

def predict_proba_to_predict(clf_classes_, y_predict_proba, y_test=None, n_accepted_probs=1):
    assert (y_test is None and n_accepted_probs == 1) or (y_test is not None and n_accepted_probs >= 1)
    ordered_classes = predict_proba_to_predict_classes(clf_classes_, y_predict_proba)
    accepted_probs = min(n_accepted_probs, len(clf_classes_))
    logger.debug("Accepted probabilities: any of the highest %s." % (accepted_probs))
    y_predict = []
    for i in range(len(ordered_classes)):
        accepted_classes = ordered_classes[i][0:accepted_probs]
        if y_test is not None and y_test[i] in accepted_classes:
            y_predict.append(y_test[i])
        else:
            y_predict.append(accepted_classes[0])
    return y_predict

def predict_proba_to_predict_classes(clf_classes_, y_predict_proba):
    from numpy import argsort, flip
    assert y_predict_proba.ndim == 2
    y_predict_classes = []
    for i in range(len(y_predict_proba)):
        ordered_classes = []
        idxs_of_sorted_higher2lower = flip(argsort(y_predict_proba[i]))
        for j in range(len(y_predict_proba[i])):
            index = idxs_of_sorted_higher2lower[j]
            classification = clf_classes_[index]
            ordered_classes.append(classification)
        y_predict_classes.append(ordered_classes)
    return y_predict_classes

def dump_json(data_frame, filename):
    f = open(filename, 'w')
    json.dump(data_frame.to_dict(orient='list'), f)
    f.close()

def generate_roc_plot(clf, X_test, y_test, filename):
    logger.debug("Generating ROC (Receiver Operating Characteristic) plot.")
    plt = plot_roc(clf, X_test, y_test)
    plt.savefig(filename)
    plt.close()
