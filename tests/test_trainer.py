import numpy as np
import pytest
from sklearn.datasets import load_digits
from tests.utils import config_file
from text_categorizer import trainer
from text_categorizer.Parameters import Parameters

def test_load_20newsgroups():
    p1 = Parameters(config_file)
    p1.excel_file = '20newsgroups'
    p2 = trainer.load_20newsgroups(p1)
    assert p1 is not p2
    assert p1 != p2
    assert p2.excel_column_with_text_data == 'data'
    assert p2.excel_column_with_classification_data == 'target'
    pass

def test_resample():
    X_train1, y_train1 = load_digits(n_class=10, return_X_y=True)
    X_train2, y_train2 = trainer.resample(None, X_train1, y_train1)
    assert np.array_equal(X_train1, X_train2)
    assert np.array_equal(y_train1, y_train2)
    X_train3, y_train3 = trainer.resample('RandomOverSample', X_train1, y_train1)
    assert X_train1.shape[0] < X_train3.shape[0]
    assert X_train1.shape[1] == X_train3.shape[1]
    for elem in X_train3:
        assert elem in X_train1
    for elem in y_train3:
        assert elem in y_train1
    X_train4, y_train4 = trainer.resample('RandomUnderSample', X_train1, y_train1)
    assert X_train1.shape[0] >= X_train4.shape[0]
    assert X_train1.shape[1] == X_train4.shape[1]
    for elem in X_train4:
        assert elem in X_train1
    for elem in y_train4:
        assert elem in y_train1
    try:
        trainer.resample('InvalidTechnique', X_train1, y_train1)
        pytest.fail()
    except ValueError as e:
        assert len(e.args) == 1
        assert e.args[0] == 'Invalid resampling method.'

def test_main():
    pass
