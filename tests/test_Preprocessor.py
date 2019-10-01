from text_categorizer.Preprocessor import Preprocessor

def test___init__():
    pass

def test_preprocess():
    pass

def test__nltk_process():
    pass

def test__signal_handler():
    pass

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
