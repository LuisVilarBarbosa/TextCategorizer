import nltk
import numpy as np
import pytest
from flair.embeddings import DocumentPoolEmbeddings
from os.path import exists, getmtime
from scipy.sparse import csr_matrix
from sklearn import datasets, feature_extraction
from tests.utils import create_temporary_file, generate_available_filename, remove_and_check
from text_categorizer import constants, pickle_manager
from text_categorizer.ContoPTParser import ContoPTParser
from text_categorizer.Document import Document
from text_categorizer.FeatureExtractor import FeatureExtractor

def test___init__():
    ft1 = FeatureExtractor()
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        pytest.fail()
    assert ft1.stop_words == set()
    assert ft1.vectorizer_file == 'vectorizer.pkl'
    assert type(ft1.vectorizer) is feature_extraction.text.TfidfVectorizer
    assert ft1.feature_reduction is None
    assert 'initial_code_to_run_on_document' in dir(ft1.document_adjustment_code)
    assert ft1.upostags_to_ignore == ['PUNCT']
    assert ft1.synonyms is None
    assert ft1.n_jobs == 1
    ft2 = FeatureExtractor(nltk_stop_words_package='english')
    assert ft2.stop_words == set(nltk.corpus.stopwords.words('english'))
    for vectorizer_name in ['CountVectorizer', 'HashingVectorizer', 'TfidfVectorizer']:
        with pytest.raises(FileNotFoundError):
            FeatureExtractor(vectorizer_name=vectorizer_name, training_mode=False, vectorizer_file=generate_available_filename())
        try:
            path = create_temporary_file(content=None, text=False)
            pickle_manager.dump(FeatureExtractor(vectorizer_name=vectorizer_name).vectorizer, path)
            ft = FeatureExtractor(vectorizer_name=vectorizer_name, training_mode=False, vectorizer_file=path)
            assert ft.vectorizer.__class__.__name__ == vectorizer_name
        finally:
            remove_and_check(path)
        ft = FeatureExtractor(vectorizer_name=vectorizer_name, training_mode=True)
        assert ft.vectorizer.__class__.__name__ == vectorizer_name
    for vectorizer_name in ['DocumentPoolEmbeddings']:
        for training_mode in [True, False]:
            vectorizer_file = generate_available_filename()
            ft = FeatureExtractor(vectorizer_name=vectorizer_name, training_mode=training_mode, vectorizer_file=vectorizer_file)
            assert ft.vectorizer.__class__.__name__ == vectorizer_name
            assert not exists(vectorizer_file)
    with pytest.raises(ValueError):
        FeatureExtractor(vectorizer_name='invalid_vectorizer', training_mode=True)
    ft3 = FeatureExtractor(remove_adjectives=True)
    assert ft3.upostags_to_ignore == ['PUNCT', 'ADJ']
    synonyms_file = 'contopt_0.1_r2_c0.0.txt'
    filename = generate_available_filename()
    try:
        ft4 = FeatureExtractor(synonyms_file=synonyms_file)
        contoPTParser = ContoPTParser(filename)
        assert ft4.synonyms == contoPTParser.synonyms
    finally:
        remove_and_check(synonyms_file)
        remove_and_check(filename)
    with pytest.raises(ValueError):
        FeatureExtractor(synonyms_file='invalid_file.txt')
    ft5 = FeatureExtractor(n_jobs=2)
    assert ft5.n_jobs == 2

def test_prepare(capsys):
    text_field = 'text field'
    class_field = 'class field'
    quantity = 2
    fields = {text_field: 'Teste value.', class_field: 'c1'}
    analyzed_sentences = {text_field: [[
        {'form': 'Teste', 'lemma': 'teste', 'upostag': None},
        {'form': 'value', 'lemma': 'value', 'upostag': None},
        {'form': '.', 'lemma': '.', 'upostag': 'PUNCT'}
    ]] * quantity}
    docs1 = [
        Document(index=0, fields=fields, analyzed_sentences=analyzed_sentences),
        Document(index=1, fields=fields, analyzed_sentences=None),
    ]
    synonyms_files = [None, 'contopt_0.1_r2_c0.0.txt']
    expected_corpus_str = [[' '.join(['teste value'] * quantity), ''], [' '.join(['prova value'] * quantity), '']]
    expected_classifications = [[fields[class_field]] * quantity] * len(synonyms_files)
    expected_idxs_to_remove = [[1]] * len(synonyms_files)
    expected_corpus = [[['teste', 'value'] * quantity, []], [['prova', 'value'] * quantity, []]]
    try:
        filename = generate_available_filename()
        pickle_manager.dump_documents(docs1, filename)
        for i, synonyms_file in enumerate(synonyms_files):
            ft = FeatureExtractor(synonyms_file=synonyms_file)
            for training_mode in [True, False]:
                corpus_str1, classifications1, idxs_to_remove1, corpus1 = ft.prepare(text_field, class_field, None, docs1, training_mode)
                corpus_str2, classifications2, idxs_to_remove2, corpus2 = ft.prepare(text_field, class_field, filename, None, training_mode)
                assert (corpus_str1, classifications1, idxs_to_remove1, corpus1) == (corpus_str2, classifications2, idxs_to_remove2, corpus2)
                assert corpus_str1 == expected_corpus_str[i]
                assert classifications1 == expected_classifications[i]
                assert idxs_to_remove1 == expected_idxs_to_remove[i]
                assert corpus1 == expected_corpus[i]
                captured = capsys.readouterr()
                assert captured.out == ''
                assert captured.err[captured.err.rfind('\r')+1:].startswith('Preparing to create classification: 100%|')
                assert captured.err.endswith('doc/s]\n') or captured.err.endswith('s/doc]\n')
            if synonyms_file is not None:
                remove_and_check(synonyms_file)
    finally:
        remove_and_check(filename)

def test__filter():
    text_field = 'text field'
    fields = {text_field: 'Teste value.', 'class field': 'c1'}
    doc = Document(index=-1, fields=fields, analyzed_sentences=None)
    upostags_to_ignore = ['PUNCT']
    assert FeatureExtractor._filter(doc, text_field, upostags_to_ignore) == []
    doc.analyzed_sentences = {text_field: [[
        {'form': 'Test', 'lemma': 'test', 'upostag': None},
        {'form': 'value', 'lemma': 'value', 'upostag': None},
        {'form': '.', 'lemma': '.', 'upostag': 'PUNCT'}
    ]] * 2}
    assert FeatureExtractor._filter(doc, text_field, upostags_to_ignore) == ['test', 'value'] * 2
    upostags_to_ignore.clear()
    assert FeatureExtractor._filter(doc, text_field, upostags_to_ignore) == ['test', 'value', '.'] * 2

def test__generate_corpus():
    lemmas = ['lemma1', 'lemma2']
    corpus = FeatureExtractor._generate_corpus(lemmas)
    assert corpus == ' '.join(lemmas)

# TODO: Test specific value of X?
def test_generate_X_y(capsys):
    quantity = 2
    corpus = ['Test lemma 1 . ' * quantity, 'Test lemma 2 . ' * quantity]
    classifications = [1, 2]
    filename = generate_available_filename()
    dpe_out = 'Please, ignore the message above indicating that the sentence is too long. The problem has been solved.\n' * 6
    combinations = [
        ('CountVectorizer', None, True, ''),
        ('CountVectorizer', 'LDA', True, ''),
        ('CountVectorizer', 'MDS', True, ''),
        ('HashingVectorizer', None, True, ''),
        ('HashingVectorizer', 'MDS', True, ''),
        ('TfidfVectorizer', None, True, ''),
        ('TfidfVectorizer', 'LDA', True, ''),
        ('TfidfVectorizer', 'MDS', True, ''),
        ('DocumentPoolEmbeddings', None, False, dpe_out),
        ('DocumentPoolEmbeddings', 'MDS', False, dpe_out),
    ]
    for vectorizer, fr, expect_file, expected_out in combinations:
        try:
            ft = FeatureExtractor(vectorizer_name=vectorizer, feature_reduction=fr, vectorizer_file=filename)
            for training_mode in [True, False]:
                assert exists(filename) is (not training_mode and expect_file)
                _X, y = ft.generate_X_y(corpus, classifications, training_mode)
                assert exists(filename) is expect_file
                assert y == classifications
                captured = capsys.readouterr()
                assert captured.out == expected_out
                assert captured.err[captured.err.rfind('\r')+1:].startswith('Extracting features: 100%|')
                assert captured.err.endswith('doc/s]\n') or captured.err.endswith('s/doc]\n')
        finally:
            if expect_file:
                remove_and_check(filename)
            if fr == 'LDA':
                remove_and_check('LatentDirichletAllocation.pkl')
    with pytest.raises(ValueError):
        FeatureExtractor(feature_reduction='invalid', vectorizer_file=filename).generate_X_y(corpus, classifications)
    remove_and_check(filename)

def test__find_incompatible_data_indexes():
    corpus = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    classifications = [20, 20, 22, 22, 24, 24, 26, 27, 28, 28]
    idxs_to_remove = FeatureExtractor._find_incompatible_data_indexes(corpus, classifications)
    assert idxs_to_remove == [6, 7]

def test_LatentDirichletAllocation():
    X = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.asarray([0, 1, 2])
    expected_new_X = np.asarray([
        [0.01428573, 0.87142845, 0.01428573, 0.01428573, 0.01428573, 0.01428573, 0.01428573, 0.01428573, 0.01428573, 0.01428573],
        [0.00625001, 0.94374995, 0.00625001, 0.00625001, 0.00625001, 0.00625001, 0.00625001, 0.00625001, 0.00625001, 0.00625001],
        [0.00400000, 0.96399997, 0.00400000, 0.00400000, 0.00400000, 0.00400000, 0.00400000, 0.00400000, 0.00400000, 0.00400000]
    ])
    filename = generate_available_filename()
    assert not exists(filename)
    try:
        new_X1, new_y1 = FeatureExtractor.LatentDirichletAllocation(X, y, filename)
        assert exists(filename)
        assert new_X1.shape == (X.shape[0], 10)
        assert np.allclose(expected_new_X, new_X1)
        assert np.array_equal(y, new_y1)
        mtime = getmtime(filename)
        new_X2, new_y2 = FeatureExtractor.LatentDirichletAllocation(X, y, filename)
        assert getmtime(filename) == mtime
        assert np.array_equal(new_X1, new_X2)
        assert np.array_equal(new_y1, new_y2)
    finally:
        remove_and_check(filename)

def test_MDS():
    X = csr_matrix(np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    y = np.asarray([0, 1, 2])
    expected_new_X = np.asarray([[0.48395794, 5.12192566], [1.00066606, -0.1960522], [-1.484624, -4.92587346]])
    new_X1, new_y1 = FeatureExtractor.MDS(X, y)
    assert new_X1.shape == (X.shape[0], 2)
    assert np.allclose(expected_new_X, new_X1)
    assert np.array_equal(y, new_y1)
    new_X2, new_y2 = FeatureExtractor.MDS(X, y)
    assert np.array_equal(new_X1, new_X2)
    assert np.array_equal(new_y1, new_y2)
