from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def create_classification():
    X, y = make_classification(n_samples=1000, n_features=4,
                                n_informative=2, n_redundant=0,
                                random_state=0, shuffle=False)
    return X, y

def random_forest_classifier():
    # https://www.kdnuggets.com/2017/12/general-approach-preprocessing-text-data.html (25/02/2019)
    X, y = create_classification()
    classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    classifier.fit(X, y)
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=2, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
                oob_score=False, random_state=0, verbose=0, warm_start=False)
    #print(classifier.feature_importances_)
    #print(classifier.predict([[0, 0, 0, 0]]))
    Xnew, _ = create_classification()
    ynew = classifier.predict_proba(Xnew)
    # show the inputs and predicted outputs
    for i in range(len(Xnew)):
    	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
