import json
import numpy as np
import os
import pandas as pd
import pytest
from sklearn.datasets import load_digits
from tests import utils
from text_categorizer import trainer
from text_categorizer.Parameters import Parameters

def test_load_20newsgroups():
    p1 = Parameters(utils.config_file)
    p1.excel_file = '20newsgroups'
    excel_file = f'{p1.excel_file}.xlsx'
    if os.path.exists(excel_file):
        utils.remove_and_check(excel_file)
    try:
        p2 = trainer.load_20newsgroups(p1)
        assert p1 is not p2
        assert p1 != p2
        assert p2.excel_column_with_text_data == 'data'
        assert p2.excel_column_with_classification_data == 'target'
        assert os.path.exists(excel_file)
        df = pd.read_excel(excel_file)
        assert df.shape == (18846, 3)
        assert list(df.keys()) == ['Unnamed: 0', 'data', 'target']
    finally:
        utils.remove_and_check(excel_file)

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

def test_dump_json():
    d1 = {'test_random_values': [np.random.random()]}
    filename = utils.generate_available_filename()
    try:
        trainer.dump_json(d1, filename)
        f = open(filename, 'r')
        d2 = json.load(f)
    finally:
        f.close()
        utils.remove_and_check(filename)
    assert d1 == d2

def test_main():
    pass
