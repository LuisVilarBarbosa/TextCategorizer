from numpy import float64, int32, int64
from tests.test_classifiers import clfs
from text_categorizer import prediction_server
from text_categorizer.FeatureExtractor import FeatureExtractor

def test_get_password():
    pass

def test_unauthorized():
    pass

def test_predict():
    pass

def test_not_found():
    pass

def test_get_feature_weights():
    corpus = ['Test corpus 1.', 'Test corpus 2.', 'Test corpus 3.']
    classifications = [1, 2, 3]
    fe = FeatureExtractor()
    X = fe.vectorizer.fit_transform(corpus)
    prediction_server._feature_extractor = fe
    lemmas = ['test', 'corpus', '1.']
    value = 0
    lemmas_dict = {'1.': value, 'corpus': value, 'test': value}
    classes_dict = {1: lemmas_dict, 2: lemmas_dict, 3: lemmas_dict}
    expected_values_1 = {
        'RandomForestClassifier': lemmas_dict,
        'BernoulliNB': classes_dict,
        'MultinomialNB': classes_dict,
        'ComplementNB': classes_dict,
        'KNeighborsClassifier': dict(),
        'MLPClassifier': dict(),
        'LinearSVC_proba': classes_dict,
        'DecisionTreeClassifier': lemmas_dict,
        'ExtraTreeClassifier': lemmas_dict,
        'DummyClassifier': dict(),
        'SGDClassifier': classes_dict,
        'BaggingClassifier': dict()
    }
    lemmas_set = {('corpus', value), ('test', value), ('1.', value), ('2.', value), ('3.', value)}
    classes_dict = {1: lemmas_set, 2: lemmas_set, 3: lemmas_set}
    expected_values_2 = {
        'RandomForestClassifier': lemmas_set,
        'BernoulliNB': classes_dict,
        'MultinomialNB': classes_dict,
        'ComplementNB': classes_dict,
        'KNeighborsClassifier': set(),
        'MLPClassifier': set(),
        'LinearSVC_proba': classes_dict,
        'DecisionTreeClassifier': lemmas_set,
        'ExtraTreeClassifier': lemmas_set,
        'DummyClassifier': set(),
        'SGDClassifier': classes_dict,
        'BaggingClassifier': set()
    }
    assert len(clfs) == len(expected_values_1)
    assert len(clfs) == len(expected_values_2)
    for f in clfs:
        clf = f(n_jobs=1, class_weight=None)
        clf_name = clf.__class__.__name__
        clf.fit(X, classifications)
        expected_value_1 = expected_values_1[clf_name]
        expected_value_2 = expected_values_2[clf_name]
        assert prediction_server._feature_weights.get(clf_name) is None
        fw1 = prediction_server.get_feature_weights(clf, lemmas)['feature_weights']
        assert replace_final_dict_values(fw1, value=value) == expected_value_1
        fw2 = prediction_server._feature_weights[clf_name]
        assert replace_tuples_values(fw2, value=value) == expected_value_2

def test_load_feature_weights():
    corpus = [['Test corpus 1.', 'Test corpus 2.'], ['Test corpus 1.', 'Test corpus 2.', 'Test corpus 3.']]
    classifications = [[1, 2], [1, 2, 3]]
    value = 0
    lemmas_sets = [{('corpus', value), ('test', value), ('1.', value), ('2.', value)}, {('corpus', value), ('test', value), ('1.', value), ('2.', value), ('3.', value)}]
    classes_dicts = [{0: lemmas_sets[0]}, {1: lemmas_sets[1], 2: lemmas_sets[1], 3: lemmas_sets[1]}]
    fe_hashing = FeatureExtractor(vectorizer_name='HashingVectorizer')
    fe_tfidf = FeatureExtractor(vectorizer_name='TfidfVectorizer')
    for i in range(2):
        X = fe_tfidf.vectorizer.fit_transform(corpus[i])
        lemmas_set = lemmas_sets[i]
        classes_dict = classes_dicts[i]
        expected_values = {
            'RandomForestClassifier': lemmas_set,
            'BernoulliNB': classes_dict,
            'MultinomialNB': classes_dict,
            'ComplementNB': classes_dict,
            'KNeighborsClassifier': set(),
            'MLPClassifier': set(),
            'LinearSVC_proba': classes_dict,
            'DecisionTreeClassifier': lemmas_set,
            'ExtraTreeClassifier': lemmas_set,
            'DummyClassifier': set(),
            'SGDClassifier': classes_dict,
            'BaggingClassifier': set()
        }
        assert len(clfs) == len(expected_values)
        for f in clfs:
            clf = f(n_jobs=1, class_weight=None)
            clf_name = clf.__class__.__name__
            clf.fit(X, classifications[i])
            prediction_server._feature_extractor = fe_hashing
            assert prediction_server.load_feature_weights(clf) == set()
            clf.fit(X, classifications[i])
            prediction_server._feature_extractor = fe_tfidf
            assert replace_tuples_values(prediction_server.load_feature_weights(clf), value=value) == expected_values[clf_name]

def test_main():
    pass

def replace_tuples_values(obj, value):
    t = type(obj)
    if t is tuple:
        return (obj[0], value)
    elif t is set:
        l = [replace_tuples_values(v, value) for v in obj]
        return set(l)
    elif t is list:
        l = [replace_tuples_values(v, value) for v in obj]
        l.sort()
        return l
    elif t is dict:
        d = dict()
        for k, v in obj.items():
            d[replace_tuples_values(k, value)] = replace_tuples_values(v, value)
        return d
    elif t is int or t is int32 or t is int64:
        return obj
    else:
        raise TypeError(t)

def replace_final_dict_values(obj, value):
    t1 = type(obj)
    if t1 is dict:
        for k, v in obj.items():
            t2 = type(v)
            if t2 is dict:
                obj[k] = replace_final_dict_values(v, value)
            elif t2 is int or t2 is int32 or t2 is int64 or t2 is float or t2 is float64:
                obj[k] = value
            else:
                raise TypeError(t2)
        return obj
    else:
        raise TypeError(t1)
