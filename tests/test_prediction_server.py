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
    lemmas = ['test', 'corpus']
    expected_values_1 = {
        'RandomForestClassifier': [('test', 0.0), ('corpus', 0.0)],
        'BernoulliNB': {1: [('test', -0.4054651081081645), ('corpus', -0.4054651081081645)], 2: [('test', -0.4054651081081645), ('corpus', -0.4054651081081645)], 3: [('test', -0.4054651081081645), ('corpus', -0.4054651081081645)]},
        'MultinomialNB': {1: [('corpus', -1.5243987931498386), ('test', -1.5243987931498386)], 2: [('corpus', -1.5243987931498386), ('test', -1.5243987931498386)], 3: [('corpus', -1.5243987931498386), ('test', -1.5243987931498386)]},
        'ComplementNB': {1: [('test', 1.4767261398842497), ('corpus', 1.4767261398842497)], 2: [('test', 1.4767261398842497), ('corpus', 1.4767261398842497)], 3: [('test', 1.4767261398842497), ('corpus', 1.4767261398842497)]},
        'KNeighborsClassifier': [],
        'MLPClassifier': [],
        'LinearSVC_proba': {1: [('test', -0.08517207986343706), ('corpus', -0.08517207986343706)], 2: [('test', -0.08517793213189749), ('corpus', -0.08517793213189749)], 3: [('test', -0.08518054004436439), ('corpus', -0.08518054004436439)]},
        'DecisionTreeClassifier': [('test', 0.0), ('corpus', 0.0)],
        'ExtraTreeClassifier': [('test', 0.0), ('corpus', 0.0)],
        'DummyClassifier': [],
        'SGDClassifier': {1: [('test', -8.888130494664434), ('corpus', -8.888130494664434)], 2: [('test', -8.88813049466443), ('corpus', -8.88813049466443)], 3: [('test', -8.888130494664434), ('corpus', -8.888130494664434)]},
        'BaggingClassifier': []
    }
    lemmas_dict = {'corpus', 'test', '1.', '2.', '3.'}
    classes_dict = {1: lemmas_dict, 2: lemmas_dict, 3: lemmas_dict}
    expected_values_2 = {
        'RandomForestClassifier': lemmas_dict,
        'BernoulliNB': classes_dict,
        'MultinomialNB': classes_dict,
        'ComplementNB': classes_dict,
        'KNeighborsClassifier': set(),
        'MLPClassifier': set(),
        'LinearSVC_proba': classes_dict,
        'DecisionTreeClassifier': lemmas_dict,
        'ExtraTreeClassifier': lemmas_dict,
        'DummyClassifier': set(),
        'SGDClassifier': classes_dict,
        'BaggingClassifier': set()
    }
    assert len(clfs) == len(expected_values_1)
    assert len(clfs) == len(expected_values_2)
    for f in clfs:
        clf = f(n_jobs=1)
        clf_name = clf.__class__.__name__
        clf.fit(X, classifications)
        expected_value_1 = expected_values_1[clf_name]
        expected_value_2 = expected_values_2[clf_name]
        assert prediction_server._feature_weights.get(clf_name) is None
        assert all(v in expected_value_1 for v in prediction_server.get_feature_weights(clf, lemmas)['feature_weights'])
        fw = prediction_server._feature_weights[clf_name]
        assert all(v in fw for v in expected_value_1)
        if type(fw) is dict:
            for k in fw.keys():
                assert len(fw[k]) > len(expected_value_1)
        else:
            assert len(fw) > len(expected_value_1) or (len(fw) == 0 and len(expected_value_1) == 0)
        assert replace_tuples_by_keys(fw) == expected_value_2

def test_load_feature_weights():
    corpus = [['Test corpus 1.', 'Test corpus 2.'], ['Test corpus 1.', 'Test corpus 2.', 'Test corpus 3.']]
    classifications = [[1, 2], [1, 2, 3]]
    lemmas_dicts = [{'corpus', 'test', '1.', '2.'}, {'corpus', 'test', '1.', '2.', '3.'}]
    classes_dicts = [{0: lemmas_dicts[0]}, {1: lemmas_dicts[1], 2: lemmas_dicts[1], 3: lemmas_dicts[1]}]
    fe_hashing = FeatureExtractor(vectorizer_name='HashingVectorizer')
    fe_tfidf = FeatureExtractor(vectorizer_name='TfidfVectorizer')
    for i in range(2):
        X = fe_tfidf.vectorizer.fit_transform(corpus[i])
        lemmas_dict = lemmas_dicts[i]
        classes_dict = classes_dicts[i]
        expected_values = {
            'RandomForestClassifier': lemmas_dict,
            'BernoulliNB': classes_dict,
            'MultinomialNB': classes_dict,
            'ComplementNB': classes_dict,
            'KNeighborsClassifier': set(),
            'MLPClassifier': set(),
            'LinearSVC_proba': classes_dict,
            'DecisionTreeClassifier': lemmas_dict,
            'ExtraTreeClassifier': lemmas_dict,
            'DummyClassifier': set(),
            'SGDClassifier': classes_dict,
            'BaggingClassifier': set()
        }
        assert len(clfs) == len(expected_values)
        for f in clfs:
            clf = f(n_jobs=1)
            clf_name = clf.__class__.__name__
            clf.fit(X, classifications[i])
            prediction_server._feature_extractor = fe_hashing
            assert prediction_server.load_feature_weights(clf) == set()
            clf.fit(X, classifications[i])
            prediction_server._feature_extractor = fe_tfidf
            assert replace_tuples_by_keys(prediction_server.load_feature_weights(clf)) == expected_values[clf_name]

def test_main():
    pass

def replace_tuples_by_keys(obj):
    if type(obj) is dict:
        d = dict()
        for k, v in obj.items():
            d[k] = set(map(lambda item: item[0], v))
        return d
    elif type(obj) is set:
        return set(map(lambda item: item[0], obj))
    else:
        raise ValueError(obj)
