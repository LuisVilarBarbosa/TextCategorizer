import json
import numpy as np
import os
import pandas as pd
import pytest
from copy import deepcopy
from sklearn.datasets import load_digits
from shutil import rmtree
from tests import utils
from text_categorizer import trainer
from text_categorizer.Parameters import Parameters

def test_load_20newsgroups():
    p1 = Parameters(utils.config_file)
    p1.excel_file = '20newsgroups'
    excel_file = utils.generate_available_filename('.xlsx')
    try:
        p2 = trainer.load_20newsgroups(p1, excel_file)
        assert p1 is not p2
        assert p1 != p2
        assert p2.excel_column_with_text_data == 'data'
        assert p2.excel_column_with_classification_data == 'target'
        assert os.path.exists(excel_file)
        df = pd.read_excel(excel_file)
        assert df.shape == (18846, 3)
        assert list(df.keys()) == ['Unnamed: 0', 'data', 'target']
        expected_mtime = os.path.getmtime(excel_file)
        p3 = trainer.load_20newsgroups(p1, excel_file)
        assert os.path.getmtime(excel_file) == expected_mtime
        assert p3.__dict__ == p2.__dict__
    finally:
        utils.remove_and_check('20news-bydate_py3.pkz')
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

@pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
def test_main():
    old_dir = os.getcwd()
    new_dir = utils.generate_available_filename()
    base_parameters = Parameters(utils.config_file)
    base_parameters.preprocessed_data_file = os.path.basename(base_parameters.preprocessed_data_file)
    try:
        os.makedirs(new_dir, exist_ok=False)
        os.chdir(new_dir)
        parameters = deepcopy(base_parameters)
        parameters.excel_file = "invalid_excel_file"
        parameters.preprocessed_data_file = "invalid_data_file"
        with pytest.raises(SystemExit):
            trainer.main(parameters)
        parameters = deepcopy(base_parameters)
        assert not os.path.exists(parameters.preprocessed_data_file)
        try:
            trainer.main(parameters)
            assert os.path.exists(parameters.preprocessed_data_file)
            assert os.path.exists("predictions.json")
            assert os.path.exists("report.xlsx")
        finally:
            utils.remove_and_check(parameters.preprocessed_data_file)
            utils.remove_and_check("predictions.json")
            utils.remove_and_check("report.xlsx")
        parameters.excel_file = os.path.abspath("20newsgroups")
        parameters.preprocess_data = False
        excel_file_20newsgroups = "20newsgroups.xlsx"
        assert not os.path.exists(excel_file_20newsgroups)
        try:
            trainer.main(parameters)
            pytest.fail()
        except SystemExit:
            assert os.path.exists(excel_file_20newsgroups)
        finally:
            utils.remove_and_check(excel_file_20newsgroups)
        parameters = deepcopy(base_parameters)
        parameters.final_training = True
        try:
            trainer.main(parameters)
        finally:
            assert not os.path.exists("predictions.json")
            assert not os.path.exists("report.xlsx")
            utils.remove_and_check(parameters.preprocessed_data_file)
    finally:
        os.chdir(old_dir)
        rmtree(new_dir)
