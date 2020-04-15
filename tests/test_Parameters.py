import pytest
from flair.embeddings import DocumentPoolEmbeddings
from multiprocessing import cpu_count
from os.path import abspath
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from text_categorizer import classifiers
from text_categorizer.Parameters import Parameters
from tests import test_classifiers, utils

def test___init__():
    expected_dict = {
        'excel_file': abspath(utils.example_excel_file),
        'excel_column_with_text_data': 'Example column',
        'excel_column_with_classification_data': 'Classification column',
        'nltk_stop_words_package': 'english',
        'number_of_jobs': cpu_count(),
        'stanfordnlp_language_package': 'en',
        'stanfordnlp_use_gpu': False,
        'stanfordnlp_resources_dir': abspath('./stanfordnlp_resources'),
        'preprocessed_data_file': abspath('./data/preprocessed_data.pkl'),
        'preprocess_data': True,
        'document_adjustment_code': abspath('./text_categorizer/document_updater.py'),
        'vectorizer': 'TfidfVectorizer',
        'feature_reduction': None,
        'set_num_accepted_probs': {1, 2, 3},
        'classifiers': test_classifiers.clfs,
        'test_subset_size': 0.3,
        'force_subsets_regeneration': False,
        'remove_adjectives': False,
        'synonyms_file': None,
        'resampling': None,
        'class_weights': None,
        'generate_roc_plots': False,
        'spell_checker_lang': None,
        'final_training': False,
        'data_dir': abspath('./data'),
    }
    parameters = Parameters(utils.config_file)
    assert parameters.__dict__ == expected_dict
    

def test__parse_number_of_jobs():
    assert Parameters._parse_number_of_jobs('None') == 1
    for i in range(1, 5):
        assert Parameters._parse_number_of_jobs(str(i)) == i
    for i in range(-4, 0):
        assert Parameters._parse_number_of_jobs(str(i)) == cpu_count() + 1 + i
    with pytest.raises(AssertionError):
        Parameters._parse_number_of_jobs(0)

def test__parse_vectorizer():
    vectorizers = [
        TfidfVectorizer.__name__,
        CountVectorizer.__name__,
        HashingVectorizer.__name__,
        DocumentPoolEmbeddings.__name__,
    ]
    for vectorizer in vectorizers:
        assert Parameters._parse_vectorizer(vectorizer) == vectorizer
    with pytest.raises(AssertionError):
        Parameters._parse_vectorizer('invalid_vectorizer')

def test__parse_accepted_probs():
    assert Parameters._parse_accepted_probs('1,2,3,2') == {1, 2, 3}
    with pytest.raises(ValueError):
        Parameters._parse_accepted_probs('')
    with pytest.raises(AssertionError):
        Parameters._parse_accepted_probs('1,0,2,3')
    with pytest.raises(AssertionError):
        Parameters._parse_accepted_probs('1,-1,2,3')

def test__parse_classifiers():
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
    clfs_str = ','.join([clf.__name__ for clf in clfs])
    assert Parameters._parse_classifiers(clfs_str) == clfs
    with pytest.raises(AssertionError):
        Parameters._parse_classifiers('')
    clfs_str = ','.join([clfs_str, 'invalid_value']) 
    with pytest.raises(AssertionError):
        Parameters._parse_classifiers(clfs_str)

def test__parse_synonyms_file():
    assert Parameters._parse_synonyms_file('None') is None
    filename = 'valid_filename'
    assert Parameters._parse_synonyms_file(filename) == abspath(filename)

def test__parse_resampling():
    assert Parameters._parse_resampling('None') is None
    values = ['RandomOverSample', 'RandomUnderSample']
    for value in values:
        assert Parameters._parse_resampling(value) is value
    with pytest.raises(AssertionError):
        Parameters._parse_resampling('invalid_value')

def test__parse_class_weights():
    assert Parameters._parse_class_weights('None') is None
    values = ['balanced']
    for value in values:
        assert Parameters._parse_class_weights(value) is value
    with pytest.raises(AssertionError):
        Parameters._parse_feature_reduction('invalid_value')

def test__parse_feature_reduction():
    assert Parameters._parse_feature_reduction('None') is None
    values = ['LDA', 'MDS']
    for value in values:
        assert Parameters._parse_feature_reduction(value) is value
    with pytest.raises(AssertionError):
        Parameters._parse_feature_reduction('invalid_value')

def test__parse_None():
    values = ['1', 2, None]
    for value in values:
        assert Parameters._parse_None(value) is value
    assert Parameters._parse_None('None') is None
