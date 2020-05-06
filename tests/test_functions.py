import json
import numpy as np
import pandas as pd
import pytest
import time
from itertools import zip_longest
from multiprocessing import cpu_count
from os.path import abspath, exists
from sys import modules
from tests import utils
from text_categorizer import functions, trainer
from text_categorizer.Parameters import Parameters

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
    df1 = pd.DataFrame()
    column_name = 'New column'
    df2 = functions.append_to_data_frame(array_2d, df1, column_name)
    with pytest.raises(KeyError):
        assert df1[column_name]
    for cell, array_1d in zip_longest(df2[column_name], array_2d):
        assert cell == ','.join(array_1d)
    with pytest.raises(ValueError):
        functions.append_to_data_frame(array_2d, df2, column_name)

def test_data_frame_to_document_list():
    df = pd.read_excel(utils.example_excel_file)
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
    expected_df1 = pd.DataFrame(data=[data1], columns=columns)
    expected_df2 = pd.DataFrame(data=[data2], columns=columns)
    try:
        path = utils.create_temporary_file(content=None, text=True)
        trainer.dump_json(predictions_dict1, path)
        f = open(path, 'r')
        predictions_dict2 = json.load(f)
        f.close()
    finally:
        utils.remove_and_check(path)
    df1 = functions.predictions_to_data_frame(predictions_dict2, 1)
    df2 = functions.predictions_to_data_frame(predictions_dict2, 2)
    assert predictions_dict1 == predictions_dict2
    pd.util.testing.assert_frame_equal(df1, expected_df1)
    pd.util.testing.assert_frame_equal(df2, expected_df2)

def test_parameters_to_data_frame():
    expected_dict = {
        'Excel file': abspath('example_excel_file.xlsx'),
        'Text column': 'Example column',
        'Label column': 'Classification column',
        'n_jobs': cpu_count(),
        'Preprocessed data file': abspath('./data/preprocessed_data.pkl'),
        'Data directory': abspath('./data'),
        'Final training': False,
        'Preprocess data': True,
        'MosesTokenizer language code': 'en',
        'Spell checker language': 'None',
        'NLTK stop words package': 'english',
        'Document adjustment code': abspath('text_categorizer/document_updater.py'),
        'Vectorizer': 'TfidfVectorizer',
        'Feature reduction': 'None',
        'Remove adjectives': False,
        'Synonyms file': 'None',
        'Accepted probabilities': {1,2,3},
        'Test size': 0.3,
        'Force subsets regeneration': False,
        'Resampling': 'None',
        'Class weights': 'None',
        'Generate ROC plots': False,
    }
    p = Parameters(utils.config_file)
    df = functions.parameters_to_data_frame(p.__dict__)
    assert df.shape == (1, 22)
    assert df.iloc[0].to_dict() == expected_dict

def test_generate_report():
    execution_info = pd.DataFrame.from_dict({
        'Start': [functions.get_local_time_str()],
        'End': [functions.get_local_time_str()],
    })
    parameters_dict = Parameters(utils.config_file).__dict__
    predictions_dict = {
        'y_true': ['label1'],
        'classifier_key': [{'label1': 0.0, 'label2': 1.0}],
    }
    parameters_dict['set_num_accepted_probs'] = 1
    expected_df_row0 = pd.concat([
        execution_info,
        functions.parameters_to_data_frame(parameters_dict),
        functions.predictions_to_data_frame(predictions_dict, 1),
    ], axis=1)
    parameters_dict['set_num_accepted_probs'] = {1}
    excel_file1 = utils.generate_available_filename(ext='.xlsx')
    excel_file2 = utils.generate_available_filename(ext='.xlsx')
    expected_df = pd.DataFrame()
    try:
        for i, file_exists in enumerate([False, True]):
            assert exists(excel_file1) is file_exists
            df = functions.generate_report(execution_info, parameters_dict, predictions_dict, excel_file1)
            df.to_excel(excel_file2, index=False)
            assert df.shape == (i + 1, 44)
            expected_df = pd.concat([expected_df, expected_df_row0])
            pd.util.testing.assert_frame_equal(df, expected_df)
            pd.util.testing.assert_frame_equal(pd.read_excel(excel_file1), pd.read_excel(excel_file2))
    finally:
        utils.remove_and_check(excel_file1)
        utils.remove_and_check(excel_file2)

def test_get_local_time_str():
    str_format = '%Y-%m-%d %H:%M:%S %z %Z'
    assert functions.get_local_time_str(time.localtime(5)) == time.strftime(str_format, time.localtime(5))
    assert pytest.approx(time.mktime(time.strptime(functions.get_local_time_str(), str_format))) == pytest.approx(time.time())
