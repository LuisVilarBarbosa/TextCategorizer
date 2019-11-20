import nltk
import pytest
import signal
from text_categorizer.Document import Document
from text_categorizer.Preprocessor import Preprocessor

def test_stop_signals():
    assert Preprocessor.stop_signals == [
        signal.SIGINT,      # SIGINT is sent by CTRL-C.
        signal.SIGTERM,     # SIGTERM is sent by Docker on CTRL-C or on a call to 'docker stop'.
    ]

def test___init__():
    p1 = Preprocessor()
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('taggers/universal_tagset')
    except LookupError:
        pytest.fail()
    assert p1.language == 'en'
    assert type(p1.lemmatizer) is nltk.stem.WordNetLemmatizer
    assert p1.stop is False
    assert p1.store_data is False
    p2 = Preprocessor(store_data=True)
    assert p2.store_data is True

def test_preprocess(capsys):
    text_field = 'Test field'
    index = -1
    fields = {text_field: 'Test value. ' * 2}
    analyzed_sentences = [[
        {'form': 'Test', 'lemma': 'test', 'upostag': None},
        {'form': 'value', 'lemma': 'value', 'upostag': None},
        {'form': '.', 'lemma': '.', 'upostag': 'PUNCT'}
    ]] * 2
    doc = Document(index=index, fields=fields, analyzed_sentences=None)
    p = Preprocessor()
    p.preprocess(text_field=text_field, preprocessed_data_file=None, docs=[doc] * 2)
    assert doc.index == index
    assert doc.fields == fields
    assert doc.analyzed_sentences == analyzed_sentences
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err[captured.err.rfind('\r')+1:].startswith('Preprocessing: 100%|')
    assert captured.err.endswith('doc/s]\n') or captured.err.endswith('s/doc]\n')

def test__nltk_process(capsys):
    text_field = 'Test field'
    index = -1
    fields = {text_field: 'Test value. ' * 2}
    analyzed_sentences = [[
        {'form': 'Test', 'lemma': 'test', 'upostag': None},
        {'form': 'value', 'lemma': 'value', 'upostag': None},
        {'form': '.', 'lemma': '.', 'upostag': 'PUNCT'}
    ]] * 2
    doc = Document(index=index, fields=fields, analyzed_sentences=None)
    p = Preprocessor()
    assert p.stop is False
    p._nltk_process(text_data_field=text_field, preprocessed_data_file=None, docs=[doc] * 2)
    assert p.stop is False
    assert doc.index == index
    assert doc.fields == fields
    assert doc.analyzed_sentences == analyzed_sentences
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err[captured.err.rfind('\r')+1:].startswith('Preprocessing: 100%|')
    assert captured.err.endswith('doc/s]\n') or captured.err.endswith('s/doc]\n')
    pass

def test__signal_handler():
    p = Preprocessor()
    assert Preprocessor.stop_signals == [signal.SIGINT, signal.SIGTERM]
    assert p.stop is False
    for sig in Preprocessor.stop_signals:
        p._signal_handler(sig=sig, frame=None)
        assert p.stop is True
        p.stop = False
    for sig in Preprocessor.stop_signals * 2:
        p._signal_handler(sig=sig, frame=None)
        assert p.stop is True
    p.stop = False
    for sig in [signal.SIGILL]:
        p._signal_handler(sig=sig, frame=None)
        assert p.stop is False

def test__set_signal_handlers():
    p = Preprocessor()
    assert 'old_handlers' not in dir(p)
    p._set_signal_handlers()
    assert len(p.old_handlers) == 2
    assert not p.stop

def test__reset_signal_handlers():
    p = Preprocessor()
    assert 'old_handlers' not in dir(p)
    p._set_signal_handlers()
    assert len(p.old_handlers) == 2
    p._reset_signal_handlers()
    assert len(p.old_handlers) == 0
    assert not p.stop
