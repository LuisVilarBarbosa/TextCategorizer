#!/usr/bin/python3
# coding=utf-8

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def random_forest_classifier(X, y):
    # The code below is based on https://ehackz.com/2018/03/23/python-scikit-learn-random-forest-classifier-tutorial/ (accessed on 2019-02-26).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # The code below is based on https://www.kdnuggets.com/2017/12/general-approach-preprocessing-text-data.html (accessed on 2019-02-25).
    clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None,
                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None,
                random_state=None, verbose=0, warm_start=False, class_weight=None)
    clf.fit(X_train, y_train)
    #print(clf.feature_importances_)
    y_predict = clf.predict(X_test)
    # The code below is based on https://ehackz.com/2018/03/23/python-scikit-learn-random-forest-classifier-tutorial/ (accessed on 2019-02-25).
    return accuracy_score(y_test, y_predict, normalize=True)
