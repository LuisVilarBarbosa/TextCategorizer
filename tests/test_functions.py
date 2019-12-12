import json
import numpy as np
import pytest
from itertools import zip_longest
from pandas import DataFrame, read_excel
from pandas.util.testing import assert_frame_equal
from sys import modules
from tests.utils import create_temporary_file, example_excel_file, remove_and_check
from text_categorizer import classifiers, functions

def test_get_python_version():
    from sys import version
    version_array = functions.get_python_version()
    assert type(version_array) is list
    for value in version_array:
        assert type(value) is int
        assert value >= 0
    obtained_version = '.'.join(str(value) for value in version_array)
    assert version.startswith(obtained_version)

def test_append_to_data_frame():
    array_2d = np.random.rand(10, 15).astype('str')
    df1 = DataFrame()
    column_name = 'New column'
    df2 = functions.append_to_data_frame(array_2d, df1, column_name)
    with pytest.raises(KeyError):
        assert df1[column_name]
    for cell, array_1d in zip_longest(df2[column_name], array_2d):
        assert cell == ','.join(array_1d)
    with pytest.raises(ValueError):
        functions.append_to_data_frame(array_2d, df2, column_name)

def test_data_frame_to_document_list():
    df = read_excel(example_excel_file)
    docs = functions.data_frame_to_document_list(df)
    assert len(docs) == len(df)
    for i in range(len(docs)):
        doc = docs[i]
        assert doc.index == i

def test_load_module():
    module_name = 'document_updater'
    assert module_name not in modules
    filename = 'text_categorizer/%s.py' % (module_name)
    module = functions.load_module(filename)
    assert 'initial_code_to_run_on_document' in dir(module)

def test_predictions_to_data_frame():
    predictions_dict1 = {
        'y_true': ['0', '1', '0', '1', '0'],
        'RandomForestClassifier': [{'0': 1., '1': 0.}, {'0': 1., '1': 0.}, \
            {'0': 0., '1': 1.}, {'0': 0., '1': 1.}, {'0': 1., '1': 0.}],
        'LinearSVC': [{'0': 1., '1': 0.}, {'0': 0., '1': 1.}, \
            {'0': 1., '1': 0.}, {'0': 0., '1': 1.}, {'0': 0., '1': 1.}]
    }
    columns = [
        '%s %s %s' % (metric, clf, label)
        for metric in ['f1-score', 'precision', 'recall', 'support']
        for clf in ['LinearSVC', 'RandomForestClassifier']
        for label in [0, 1, 'macro avg', 'micro avg', 'weighted avg']
    ]
    data1 = [
        0.8, 0.8, 0.8, 0.8000000000000002, 0.8,
        0.6666666666666666, 0.5, 0.5833333333333333, 0.6, 0.6,
        1.0, 0.6666666666666666, 0.8333333333333333, 0.8, 0.8666666666666666,
        0.6666666666666666, 0.5, 0.5833333333333333, 0.6, 0.6,
        0.6666666666666666, 1.0, 0.8333333333333333, 0.8, 0.8,
        0.6666666666666666, 0.5, 0.5833333333333333, 0.6, 0.6,
        3, 2, 5, 5, 5,
        3, 2, 5, 5, 5
    ]
    data2 = data1.copy()
    data2[0:30] = [1.] * 30
    expected_df1 = DataFrame(data=[data1], columns=columns)
    expected_df2 = DataFrame(data=[data2], columns=columns)
    path = create_temporary_file(content=None, text=True)
    classifiers.dump_json(predictions_dict1, path)
    f = open(path, 'r')
    predictions_dict2 = json.load(f)
    f.close()
    remove_and_check(path)
    df1 = functions.predictions_to_data_frame(predictions_dict2, 1)
    df2 = functions.predictions_to_data_frame(predictions_dict2, 2)
    assert predictions_dict1 == predictions_dict2
    assert_frame_equal(df1, expected_df1)
    assert_frame_equal(df2, expected_df2)

def test_parameters_to_data_frame():
    pass

def test_generate_report():
    pass

def test_get_local_time_str():
    pass
