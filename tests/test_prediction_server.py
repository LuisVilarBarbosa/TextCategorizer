import pytest
from base64 import b64encode
from multiprocessing import cpu_count
from numpy import float64, int32, int64
from os.path import abspath
from pandas import read_excel
from tests.test_classifiers import clfs
from tests import utils
from text_categorizer import prediction_server
from text_categorizer.FeatureExtractor import FeatureExtractor
from text_categorizer.functions import data_frame_to_document_list
from text_categorizer.Parameters import Parameters
from text_categorizer.pickle_manager import dump
from text_categorizer.Preprocessor import Preprocessor

valid_headers = {'Authorization': 'Basic {0}'.format(utils.decode(b64encode(b'admin:admin')))}
invalid_headers = {'Authorization': 'Basic {0}'.format(utils.decode(b64encode(b'username:password')))}

@pytest.fixture
def client():
    with prediction_server.app.test_client() as client:
        yield client

def test_get_password(client):
    res = client.post('/')
    assert res.status_code == prediction_server.UNAUTHORIZED_ACCESS
    res = client.post('/', headers=valid_headers)
    assert res.status_code == prediction_server.BAD_REQUEST
    res = client.post('/', json={'text': 'Test text.', 'classifier': 'LinearSVC'}, headers=valid_headers)
    assert res.status_code in [200, 500]
    res = client.post('/', headers=invalid_headers)
    assert res.status_code == prediction_server.UNAUTHORIZED_ACCESS

def test_unauthorized(client):
    res = client.post('/', headers=invalid_headers)
    assert res.status_code == prediction_server.UNAUTHORIZED_ACCESS
    assert res.json == {'error': 'Unauthorized access'}

def test_predict(client):
    df = read_excel(utils.example_excel_file)
    docs = data_frame_to_document_list(df)
    prediction_server._text_field = 'Example column'
    prediction_server._class_field = 'Classification column'
    clfs_filenames = []
    try:
        vectorizer_path = utils.create_temporary_file(content=None, text=False)
        p = Preprocessor()
        p.preprocess(text_field=prediction_server._text_field, preprocessed_data_file=None, docs=docs)
        ft = FeatureExtractor(training_mode=True, vectorizer_file=vectorizer_path)
        corpus, classifications, _, _ = ft.prepare(text_field=prediction_server._text_field, class_field=prediction_server._class_field, preprocessed_data_file=None, docs=docs, training_mode=True)
        X, y = ft.generate_X_y(corpus, classifications, training_mode=True)
        prediction_server._preprocessor = Preprocessor()
        prediction_server._feature_extractor = FeatureExtractor(training_mode=False, vectorizer_file=vectorizer_path)
        res = client.post('/', headers=valid_headers)
        assert res.status_code == prediction_server.BAD_REQUEST
        res = client.post('/', json={'text': 1, 'classifier': 'LinearSVC'}, headers=valid_headers)
        assert res.status_code == prediction_server.BAD_REQUEST
        assert utils.decode(res.data).endswith('<p>Invalid text</p>\n')
        res = client.post('/', json={'text': 'Test text.', 'classifier': 1}, headers=valid_headers)
        assert res.status_code == prediction_server.BAD_REQUEST
        assert utils.decode(res.data).endswith('<p>Invalid classifier</p>\n')
        res = client.post('/', json={'text': 'Test text.', 'classifier': '../LinearSVC'}, headers=valid_headers)
        assert res.status_code == prediction_server.BAD_REQUEST
        assert utils.decode(res.data).endswith('<p>Invalid classifier</p>\n')
        res = client.post('/', json={'text': 'Test text.', 'classifier': 'LinearSVC'}, headers=valid_headers)
        assert res.status_code == prediction_server.BAD_REQUEST
        assert utils.decode(res.data).endswith('<p>Invalid classifier model</p>\n')
        for f in clfs:
            clf_filename_base = utils.generate_available_filename()
            clf_filename = '%s.pkl' % (clf_filename_base)
            clfs_filenames.append(clf_filename)
            clf = f(n_jobs=1, class_weight=None)
            clf.fit(X, y)
            dump(clf, clf_filename)
            res = client.post('/', json={'text': 'Test text.', 'classifier': clf_filename_base},    headers=valid_headers)
            assert res.status_code == 200
            assert repr(prediction_server._classifiers[clf_filename_base]) == repr(clf)
            assert replace_final_dict_values(res.json, value=0) in [
                {'feature_weights': {'I': {}, 'II': {}, 'III': {}}, 'probabilities': {'I': 0, 'II': 0, 'III': 0}},
                {'feature_weights': {}, 'probabilities': {'I': 0, 'II': 0, 'III': 0}}
            ]
    finally:
        utils.remove_and_check(vectorizer_path)
        for clf_filename in clfs_filenames:
            utils.remove_and_check(clf_filename)
        prediction_server._text_field = None
        prediction_server._class_field = None
        prediction_server._preprocessor = None
        prediction_server._feature_extractor = None
        prediction_server._feature_weights = dict()
        prediction_server._classifiers = dict()

def test_not_found(client):
    res = client.post('/invalid_target', headers=invalid_headers)
    assert res.status_code == prediction_server.NOT_FOUND
    assert res.json == {'error': 'Not found'}

@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
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

@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
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

def test_main(monkeypatch):
    parameters = Parameters(utils.config_file)
    with pytest.raises(SystemExit):
        prediction_server.main(parameters, 1024)
    with monkeypatch.context() as m:
        m.setattr("text_categorizer.prediction_server.app.run", lambda host, port, debug: None)
        try:
            vectorizer_file = 'vectorizer.pkl'
            dump(FeatureExtractor(vectorizer_name='TfidfVectorizer').vectorizer, vectorizer_file)
            assert not prediction_server.logger.disabled
            prediction_server.main(parameters, 1025)
            assert prediction_server.logger.disabled
            assert prediction_server._text_field == 'Example column'
            assert prediction_server._class_field == 'Classification column'
            assert prediction_server._preprocessor.language == 'en'
            assert prediction_server._preprocessor.store_data is False
            assert prediction_server._preprocessor.spell_checker is None
            #assert prediction_server._preprocessor.spell_checker.hunspell.max_threads == cpu_count()
            assert len(prediction_server._feature_extractor.stop_words) > 0
            assert prediction_server._feature_extractor.feature_reduction is None
            assert prediction_server._feature_extractor.document_adjustment_code.__file__ == abspath('text_categorizer/document_updater.py')
            assert prediction_server._feature_extractor.synonyms is None
            assert prediction_server._feature_extractor.vectorizer_file == vectorizer_file
            assert prediction_server._feature_extractor.n_jobs == cpu_count()
        finally:
            utils.remove_and_check(vectorizer_file)

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
