import numpy as np
import pytest
from itertools import zip_longest
from pandas import DataFrame, read_excel
from sys import modules
from text_categorizer import functions
from text_categorizer.Document import _excel_start_row
from tests.utils import example_excel_file

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
        assert doc.index == i + _excel_start_row

def test_load_module():
    module_name = 'document_updater'
    assert module_name not in modules
    filename = 'text_categorizer/%s.py' % (module_name)
    module = functions.load_module(filename)
    assert 'initial_code_to_run_on_document' in dir(module)
