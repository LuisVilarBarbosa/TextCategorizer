#!/usr/bin/python3
# coding=utf-8

import pickle_manager

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from logger import logger

def RandomForestClassifier(n_jobs):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None,
                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                min_impurity_split=None, bootstrap=True, oob_score=False,
                n_jobs=n_jobs, random_state=None, verbose=0,
                warm_start=False, class_weight=None)
    return clf

def BernoulliNB(n_jobs):
    from sklearn.naive_bayes import BernoulliNB
    clf = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
    return clf

def MultinomialNB(n_jobs):
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    return clf

def ComplementNB(n_jobs):
    from sklearn.naive_bayes import ComplementNB
    clf = ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)
    return clf

def KNeighborsClassifier(n_jobs):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto',
                leaf_size=30, p=2, metric='minkowski', metric_params=None,
                n_jobs=n_jobs)
    return clf

def MLPClassifier(n_jobs):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam',
                alpha=0.0001, batch_size='auto', learning_rate='constant',
                learning_rate_init=0.001, power_t=0.5, max_iter=300, shuffle=True, # Predefined max_iter: 200
                random_state=None, tol=0.0001, verbose=False, warm_start=False,
                momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                n_iter_no_change=10)
    return clf

def SVC(n_jobs):
    from sklearn.svm import SVC
    clf = SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0,
                shrinking=True, probability=True, tol=0.0001, cache_size=200,
                class_weight=None, verbose=False, max_iter=-1,
                decision_function_shape='ovr', random_state=None)
    return clf

def DecisionTreeClassifier(n_jobs):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None,
                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                max_features=None, random_state=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None,
                presort=False)
    return clf

def ExtraTreeClassifier(n_jobs):
    from sklearn.tree import ExtraTreeClassifier
    clf = ExtraTreeClassifier(criterion='gini', splitter='random', max_depth=None,
                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                max_features='auto', random_state=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None)
    return clf

def DummyClassifier(n_jobs):
    from sklearn.dummy import DummyClassifier
    clf = DummyClassifier(strategy='stratified', random_state=None, constant=None)
    return clf

def SGDClassifier(n_jobs):
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.0001, l1_ratio=0.15,
                fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True,
                verbose=0, epsilon=0.1, n_jobs=n_jobs,
                random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5,
                early_stopping=False, validation_fraction=0.1, n_iter_no_change=5,
                class_weight=None, warm_start=False, average=False, n_iter=None)
    return clf

def BaggingClassifier(n_jobs):
    from sklearn.ensemble import BaggingClassifier
    clf = BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0,
                max_features=1.0, bootstrap=True, bootstrap_features=False,
                oob_score=False, warm_start=False, n_jobs=n_jobs,
                random_state=None, verbose=0)
    return clf

def RFE(n_jobs): # TODO: Allow to choose the estimator through the configuration file.
    from sklearn.feature_selection import RFE
    selector = RFE(estimator=ComplementNB(n_jobs), n_features_to_select=None, step=1, verbose=0)
    return selector

def RFECV(n_jobs): # TODO: Allow to choose the estimator through the configuration file.
    from sklearn.feature_selection import RFECV
    selector = RFECV(estimator=ComplementNB(n_jobs=1), step=1, min_features_to_select=1, cv=5, scoring=None,
                verbose=0, n_jobs=n_jobs)
    return selector

class Pipeline():
    def __init__(self, classifiers, cross_validate):
        self.classifiers = classifiers
        self.cross_validate = cross_validate
        logger.debug("Cross validate = %s" % (self.cross_validate))
    
    def start(self, X, y, n_jobs=None):
        from sklearn.model_selection import cross_validate
        from time import time
        if not self.cross_validate:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)
        for f in self.classifiers:
            logger.info("Starting %s." % (f.__name__))
            clf = f(n_jobs=n_jobs)
            logger.debug("%s configuration: %s" % (f.__name__, clf.__dict__))
            t1 = time()
            try:
                if self.cross_validate:
                    scores = cross_validate(estimator=clf, X=X, y=y, groups=None, scoring=None, cv=5,
                                n_jobs=n_jobs, verbose=0,
                                fit_params=None, pre_dispatch='2*n_jobs',
                                return_train_score='warn', return_estimator=False,
                                error_score='raise')
                    out = scores
                else:
                    clf.fit(X_train, y_train)
                    clf_filename = "%s.pkl" % (clf.__class__.__name__)
                    pickle_manager.dump(clf, clf_filename)
                    y_predict_proba = clf.predict_proba(X_test)
                    y_predict = predict_proba_to_predict(clf.classes_, y_predict_proba, y_test)
                    logger.debug("Confusion matrix:\n%s" % confusion_matrix(y_test, y_predict))
                    logger.debug("Classification report:\n%s" % classification_report(y_test, y_predict))
                    out = accuracy_score(y_test, y_predict, normalize=True)
                logger_func = logger.info
            except Exception as e:
                out = repr(e)
                logger_func = logger.error
            logger_func("%s: %s | %ss" % (f.__name__, out, (time() - t1)))

def predict_proba_to_predict(clf_classes_, y_predict_proba, y_test=None):
    ordered_classes = predict_proba_to_predict_classes(clf_classes_, y_predict_proba)
    accepted_probs = min(1, len(clf_classes_))
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
