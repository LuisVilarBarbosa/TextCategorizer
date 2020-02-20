import numpy as np
from collections import Counter
from itertools import zip_longest
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
    
    def start(self, X_train, y_train, X_test=np.array([]), y_test=[], n_jobs=None, set_n_accepted_probs={1,2,3}, class_weight=None, generate_roc_plots=False):
        assert (X_test.shape[0] > 0 and y_test) or (X_test.shape[0] == 0 and not y_test) 
        logger.debug("Number of training examples: %s" % (Counter(y_train)))
        predictions = {'y_true': y_test}
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
                clf.fit(X_train, y_train)
                pickle_manager.dump(clf, clf_filename)
                if X_test.shape[0] > 0 and y_test:
                    y_predict_proba = clf.predict_proba(X_test)
                    dicts = predict_proba_to_dicts(clf.classes_, y_predict_proba)
                    predictions[f.__name__] = dicts
                    for n_accepted_probs in set_n_accepted_probs:
                        logger.debug("Accepted probabilities: any of the highest %s." % (n_accepted_probs))
                        y_predict = dicts_to_predict(dicts, y_test, n_accepted_probs)
                        if n_accepted_probs == 1 and generate_roc_plots:
                            generate_roc_plot(clf, X_test, y_test, 'ROC_%s.png' % (f.__name__))
                        logger.debug("Confusion matrix:\n%s" % confusion_matrix(y_test, y_predict))
                        logger.debug("Classification report:\n%s" % classification_report(y_test, y_predict))
                        acc = accuracy_score(y_test, y_predict, normalize=True)
                        logger.info("%s: %s | %ss" % (f.__name__, acc, (time() - t1)))
                else:
                    logger.info("%s: %s | %ss" % (f.__name__, 'Finished', (time() - t1)))
            except Exception as e:
                logger.error("%s: %s | %ss" % (f.__name__, repr(e), (time() - t1)))
        return predictions

def predict_proba_to_dicts(clf_classes_, y_predict_proba):
    assert len(clf_classes_) == y_predict_proba.shape[1]
    my_clf_classes_ = clf_classes_.tolist()
    my_y_predict_proba = y_predict_proba.tolist()
    my_y_predict_proba = [dict(zip_longest(my_clf_classes_, probs)) for probs in my_y_predict_proba]
    return my_y_predict_proba

def dicts_to_predict(dicts, y_true=None, n_accepted_probs=1):
    assert (y_true is None and n_accepted_probs == 1) \
        or (y_true is not None and n_accepted_probs >= 1 and len(dicts) == len(y_true))
    sorted_probs = [sorted(d.items(), key=lambda item: -item[1]) for d in dicts]
    accepted_classes = [[item[0] for item in l[0:n_accepted_probs]] for l in sorted_probs]
    if y_true is None:
        y_pred = [cs[0] for cs in accepted_classes]
    else:
        y_pred = [t if t in cs else cs[0] for cs, t in zip_longest(accepted_classes, y_true)]
    return y_pred

def generate_roc_plot(clf, X_test, y_test, filename):
    logger.debug("Generating ROC (Receiver Operating Characteristic) plot.")
    plt = plot_roc(clf, X_test, y_test)
    plt.savefig(filename)
    plt.close()
