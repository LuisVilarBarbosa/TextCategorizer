#!/usr/bin/python3
# coding=utf-8

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def RandomForestClassifier(X, y):
    from sklearn.ensemble import RandomForestClassifier
    # The code below is based on https://ehackz.com/2018/03/23/python-scikit-learn-random-forest-classifier-tutorial/ (accessed on 2019-02-26).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)
    # The code below is based on https://www.kdnuggets.com/2017/12/general-approach-preprocessing-text-data.html (accessed on 2019-02-25).
    clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None,
                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None,
                random_state=None, verbose=0, warm_start=False, class_weight=None)
    clf.fit(X_train, y_train)
    #print(clf.feature_importances_)
    y_predict = clf.predict(X_test)
    # The code below is based on https://stackabuse.com/text-classification-with-python-and-scikit-learn/ (accessed on 2019-02-26).
    #from sklearn.metrics import confusion_matrix, classification_report
    #print(confusion_matrix(y_test, y_predict))  
    #print(classification_report(y_test, y_predict))
    # The code below is based on https://ehackz.com/2018/03/23/python-scikit-learn-random-forest-classifier-tutorial/ (accessed on 2019-02-25).
    return accuracy_score(y_test, y_predict, normalize=True)

def BernoulliNB(X, y):
    from sklearn.naive_bayes import BernoulliNB
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)
    clf = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    return accuracy_score(y_test, y_predict, normalize=True)

def GaussianNB(X, y):
    from sklearn.naive_bayes import GaussianNB
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)
    clf = GaussianNB(priors=None, var_smoothing=1e-09)
    clf.fit(X_train.toarray(), y_train)
    y_predict = clf.predict(X_test.toarray())
    return accuracy_score(y_test, y_predict, normalize=True)

def MultinomialNB(X, y):
    from sklearn.naive_bayes import MultinomialNB
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)
    clf = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    return accuracy_score(y_test, y_predict, normalize=True)

def ComplementNB(X, y):
    from sklearn.naive_bayes import ComplementNB
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)
    clf = ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    return accuracy_score(y_test, y_predict, normalize=True)

def KNeighborsClassifier(X, y):
    from sklearn.neighbors import KNeighborsClassifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)
    clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto',
                leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    return accuracy_score(y_test, y_predict, normalize=True)

def BernoulliRBM(X, y):
    from sklearn.neural_network import BernoulliRBM
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)
    clf = BernoulliRBM(n_components=256, learning_rate=0.1, batch_size=10, n_iter=10,
                verbose=0, random_state=None)
    clf.fit(X_train, y_train)
    return clf.score_samples(X_test)

def MLPClassifier(X, y):
    from sklearn.neural_network import MLPClassifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)
    clf = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam',
                alpha=0.0001, batch_size='auto', learning_rate='constant',
                learning_rate_init=0.001, power_t=0.5, max_iter=300, shuffle=True, # Predefined max_iter: 200
                random_state=None, tol=0.0001, verbose=False, warm_start=False,
                momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                n_iter_no_change=10)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    return accuracy_score(y_test, y_predict, normalize=True)

def LinearSVC(X, y):
    from sklearn.svm import LinearSVC
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)
    clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0,
                multi_class='ovr', fit_intercept=True, intercept_scaling=1,
                class_weight=None, verbose=0, random_state=None, max_iter=1000)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    return accuracy_score(y_test, y_predict, normalize=True)

def NuSVC(X, y):
    from sklearn.svm import NuSVC
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)
    clf = NuSVC(nu=0.5, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0,
                shrinking=True, probability=False, tol=0.001, cache_size=200,
                class_weight=None, verbose=False, max_iter=-1,
                decision_function_shape='ovr', random_state=None)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    return accuracy_score(y_test, y_predict, normalize=True)

def DecisionTreeClassifier(X, y):
    from sklearn.tree import DecisionTreeClassifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)
    clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None,
                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                max_features=None, random_state=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None,
                presort=False)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    return accuracy_score(y_test, y_predict, normalize=True)

def ExtraTreeClassifier(X, y):
    from sklearn.tree import ExtraTreeClassifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)
    clf = ExtraTreeClassifier(criterion='gini', splitter='random', max_depth=None,
                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                max_features='auto', random_state=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    return accuracy_score(y_test, y_predict, normalize=True)

def ClassifierMixin(X, y):
    from sklearn.base import ClassifierMixin
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)
    clf = ClassifierMixin()
    return clf.score(X_train, y_train)

class Pipeline():
    def __init__(self, classifiers):
        self.classifiers = classifiers
    
    def start(self, X, y):
        from time import time
        from logger import logger
        for f in self.classifiers:
            t1 = time()
            try:
                out = f(X, y)
            except MemoryError:
                out = MemoryError.__name__
            logger.info("%s: %s | %ss" % (f.__name__, out, (time() - t1)))
