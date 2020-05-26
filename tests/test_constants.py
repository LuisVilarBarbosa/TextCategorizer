import signal
from text_categorizer import constants

def test_stop_signals():
    assert constants.stop_signals == [
        #signal.SIGINT,
        signal.SIGTERM,
    ]

def test_random_state():
    assert constants.random_state == 42
