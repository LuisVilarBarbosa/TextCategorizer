import io
import nltk
import pytest
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from tests.utils import create_temporary_file, remove_and_check
from text_categorizer.ContoPTParser import ContoPTParser
from text_categorizer.Document import Document
from text_categorizer.FeatureExtractor import FeatureExtractor
from zipfile import ZipFile

def test___init__():
    ft1 = FeatureExtractor()
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        pytest.fail()
    assert ft1.stop_words == set()
    assert ft1.vectorizer_file == 'vectorizer.pkl'
    assert type(ft1.vectorizer) is TfidfVectorizer
    assert ft1.feature_reduction is None
    assert 'initial_code_to_run_on_document' in dir(ft1.document_adjustment_code)
    assert ft1.upostags_to_ignore == ['PUNCT']
    assert ft1.synonyms is None
    ft2 = FeatureExtractor(nltk_stop_words_package='english')
    assert ft2.stop_words == set(nltk.corpus.stopwords.words('english'))
    ft3 = FeatureExtractor(remove_adjectives=True)
    assert ft3.upostags_to_ignore == ['PUNCT', 'ADJ']
    r = requests.get(url='http://ontopt.dei.uc.pt/recursos/CONTO.PT.01.zip', stream=True)
    with ZipFile(io.BytesIO(r.content)) as archive:
        data = archive.read('contopt_0.1_r2_c0.0.txt')
        try:
            path = create_temporary_file(content=data, text=False)
            ft4 = FeatureExtractor(synonyms_file=path)
            contoPTParser = ContoPTParser(path)
        finally:
            remove_and_check(path)
        assert ft4.synonyms == contoPTParser.synonyms

def test_prepare():
    pass

def test__filter():
    doc = Document(index=-1, fields={}, analyzed_sentences=None)
    upostags_to_ignore = ['PUNCT']
    assert FeatureExtractor._filter(doc, upostags_to_ignore) == []
    doc.analyzed_sentences = [[
        {'form': 'Test', 'lemma': 'test', 'upostag': None},
        {'form': 'value', 'lemma': 'value', 'upostag': None},
        {'form': '.', 'lemma': '.', 'upostag': 'PUNCT'}
    ]] * 2
    assert FeatureExtractor._filter(doc, upostags_to_ignore) == ['test', 'value'] * 2
    upostags_to_ignore.clear()
    assert FeatureExtractor._filter(doc, upostags_to_ignore) == ['test', 'value', '.'] * 2

def test__generate_corpus():
    lemmas = ['lemma1', 'lemma2']
    corpus = FeatureExtractor._generate_corpus(lemmas)
    assert corpus == ' '.join(lemmas)

def test_generate_X_y():
    pass

def test__find_incompatible_data_indexes():
    corpus = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    classifications = [20, 20, 22, 22, 24, 24, 26, 27, 28, 28]
    idxs_to_remove = FeatureExtractor._find_incompatible_data_indexes(corpus, classifications)
    assert idxs_to_remove == [6, 7]

def test__LatentDirichletAllocation():
    pass

def test__get_vectorizer():
    pass

def test_chunked_embed():
    pass

def test__MDS():
    pass
