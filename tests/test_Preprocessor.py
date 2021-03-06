import nltk
import pytest
import signal
from pandas import read_excel
from shutil import rmtree
from tests import utils
from text_categorizer import constants, functions, pickle_manager
from text_categorizer.Document import Document
from text_categorizer.Preprocessor import Preprocessor

def test___init__():
    p1 = Preprocessor()
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        pytest.fail()
    assert p1.mosestokenizer_language_code == 'en'
    assert type(p1.lemmatizer) is nltk.stem.WordNetLemmatizer
    assert p1.stop is False
    assert p1.store_data is False
    assert p1.spell_checker is None
    p2 = Preprocessor(store_data=True)
    assert p2.store_data is True
    try:
        p3 = Preprocessor(spell_checker_lang='en_US')
        assert p3.spell_checker.hunspell.lang == 'en_US'
        assert p3.spell_checker.hunspell.max_threads == 1
        del(p3)
        p4 = Preprocessor(spell_checker_lang='en_US', n_jobs=2)
        assert p4.spell_checker.hunspell.max_threads == 2
        del(p4)
    finally:
        rmtree('./hunspell')

def test_preprocess(capsys):
    text_field = 'Test field'
    index = -1
    fields = {text_field: 'Teste\r\nvalue with\ra\nfew tikens. ' * 2}
    analyzed_sentences1 = {text_field: [[
        {'form': 'Teste', 'lemma': 'teste', 'upostag': None},
        {'form': 'value', 'lemma': 'value', 'upostag': None},
        {'form': 'with', 'lemma': 'with', 'upostag': None},
        {'form': 'a', 'lemma': 'a', 'upostag': None},
        {'form': 'few', 'lemma': 'few', 'upostag': None},
        {'form': 'tikens', 'lemma': 'tikens', 'upostag': None},
        {'form': '.', 'lemma': '.', 'upostag': 'PUNCT'}
    ]] * 2}
    analyzed_sentences2 = {text_field: [[
        {'form': 'Test', 'lemma': 'test', 'upostag': None},
        {'form': 'value', 'lemma': 'value', 'upostag': None},
        {'form': 'with', 'lemma': 'with', 'upostag': None},
        {'form': 'a', 'lemma': 'a', 'upostag': None},
        {'form': 'few', 'lemma': 'few', 'upostag': None},
        {'form': 'tokens', 'lemma': 'token', 'upostag': None},
        {'form': '.', 'lemma': '.', 'upostag': 'PUNCT'}
    ]] * 2}
    for spell_checker_lang, analyzed_sentences in [(None, analyzed_sentences1), ('en_US', analyzed_sentences2)]:
        doc = Document(index=index, fields=fields, analyzed_sentences=dict())
        p = Preprocessor(spell_checker_lang=spell_checker_lang)
        assert p.stop is False
        p.preprocess(text_field=text_field, preprocessed_data_file=None, docs=[doc] * 2)
        assert p.stop is False
        assert doc.index == index
        assert doc.fields == fields
        assert doc.analyzed_sentences == analyzed_sentences
        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err[captured.err.rfind('\r')+1:].startswith('Preprocessing: 100%|')
        assert captured.err.endswith('doc/s]\n') or captured.err.endswith('s/doc]\n')
        p.stop = True
        with pytest.raises(SystemExit):
            p.preprocess(text_field=text_field, preprocessed_data_file=None, docs=[doc] * 2)
        del(p)
        if spell_checker_lang is not None:
            rmtree('./hunspell')
    docs = [Document(index=index, fields=fields, analyzed_sentences=dict()) for index in range(2)]
    preprocessed_data_file = utils.generate_available_filename()
    try:
        pickle_manager.dump_documents(docs, preprocessed_data_file)
        pickle_manager.check_data(preprocessed_data_file)
        p = Preprocessor(store_data=True)
        assert all([doc.analyzed_sentences == dict() for doc in pickle_manager.get_documents(preprocessed_data_file)])
        p.preprocess(text_field, preprocessed_data_file, None)
        assert all([doc.analyzed_sentences == analyzed_sentences1 for doc in pickle_manager.get_documents(preprocessed_data_file)])
        pickle_manager.check_data(preprocessed_data_file)
    finally:
        utils.remove_and_check(preprocessed_data_file)

def test__signal_handler():
    p = Preprocessor()
    assert p.stop is False
    for sig in constants.stop_signals:
        p._signal_handler(sig=sig, frame=None)
        assert p.stop is True
        p.stop = False
    for sig in constants.stop_signals * 2:
        p._signal_handler(sig=sig, frame=None)
        assert p.stop is True
    p.stop = False
    for sig in [signal.SIGILL]:
        assert sig not in constants.stop_signals
        p._signal_handler(sig=sig, frame=None)
        assert p.stop is False

def test__set_signal_handlers():
    p = Preprocessor()
    assert 'old_handlers' not in dir(p)
    p._set_signal_handlers()
    assert len(p.old_handlers) == 1
    assert not p.stop

def test__reset_signal_handlers():
    p = Preprocessor()
    assert 'old_handlers' not in dir(p)
    p._set_signal_handlers()
    assert len(p.old_handlers) == 1
    p._reset_signal_handlers()
    assert len(p.old_handlers) == 0
    assert not p.stop
