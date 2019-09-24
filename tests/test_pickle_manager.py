import pytest
from itertools import zip_longest
from os.path import exists, getctime
from pandas import read_excel
from pickle import load
from random import random
from text_categorizer import pickle_manager
from text_categorizer.functions import data_frame_to_document_list
from tests.utils import create_temporary_file, example_excel_file, generate_available_filename, remove_and_check
from time import time

def test__pickle_protocol():
    assert pickle_manager._pickle_protocol == 2

def test_dump_and_load():
    obj1 = random()
    path = create_temporary_file(content=None, text=False)
    pickle_manager.dump(obj1, path)
    obj2 = pickle_manager.load(path)
    assert obj1 == obj2
    remove_and_check(path)

def test_get_documents():
    df = read_excel(example_excel_file)
    docs1 = data_frame_to_document_list(df)
    filename = generate_available_filename()
    pickle_manager.dump_documents(docs1, filename)
    docs2 = pickle_manager.get_documents(filename)
    for doc1, doc2 in zip_longest(docs1, docs2):
        assert repr(doc1) == repr(doc2)
    remove_and_check(filename)

def test_dump_documents():
    df = read_excel(example_excel_file)
    docs1 = data_frame_to_document_list(df)
    path = create_temporary_file(content=None, text=False)
    with pytest.raises(Exception):
        pickle_manager.dump_documents(docs1, path)
    remove_and_check(path)
    filename = generate_available_filename()
    pickle_manager.dump_documents(docs1, filename)
    metadata = pickle_manager.get_docs_metadata(filename)
    docs2 = pickle_manager.get_documents(filename)
    assert len(metadata) == 1
    assert metadata['total'] == len(docs1)
    for doc1, doc2 in zip_longest(docs1, docs2):
        assert repr(doc1) == repr(doc2)
    remove_and_check(filename)

def test_check_data():
    df = read_excel(example_excel_file)
    docs = data_frame_to_document_list(df)
    filename = generate_available_filename()
    pickle_manager.dump_documents(docs, filename)
    pickle_manager.check_data(filename)
    count = 10
    metadata1 = {'total': count}
    pda1 = pickle_manager.PickleDumpAppend(metadata1, filename)
    for not_Document in range(count):
        pda1.dump_append(not_Document)
    pda1.close()
    with pytest.raises(AssertionError):
        pickle_manager.check_data(filename)
    metadata2 = {'total': -1}
    pickle_manager.PickleDumpAppend(metadata2, filename).close()
    with pytest.raises(AssertionError):
        pickle_manager.check_data(filename)
    remove_and_check(filename)

def test__generate_file():
    filename = pickle_manager._generate_file()
    assert exists(filename)
    assert pytest.approx(getctime(filename)) == pytest.approx(time())
    remove_and_check(filename)

def test_get_docs_metadata():
    df = read_excel(example_excel_file)
    docs = data_frame_to_document_list(df)
    filename = generate_available_filename()
    pickle_manager.dump_documents(docs, filename)
    metadata = pickle_manager.get_docs_metadata(filename)
    assert type(metadata) is dict
    assert len(metadata) == 1
    assert metadata['total'] == len(docs)
    remove_and_check(filename)


def test_PickleDumpAppend___init__():
    metadata = {'total': 0}
    filename = generate_available_filename()
    not_dict = 'test_str'
    not_str = -1
    params = [[not_dict, filename], [metadata, not_str]]
    for m, f in params:
        with pytest.raises(AssertionError):
            pda = pickle_manager.PickleDumpAppend(m, f)
    pda = pickle_manager.PickleDumpAppend(metadata, filename)
    assert pda.filename_upon_completion == filename
    assert exists(pda.file.name)
    pda.close()
    assert pickle_manager.load(filename) == metadata
    assert not exists(pda.file.name)
    assert exists(filename)
    remove_and_check(filename)

def test_PickleDumpAppend_dump_append():
    count = 10
    metadata = {'total': 0}
    filename = generate_available_filename()
    pda = pickle_manager.PickleDumpAppend(metadata, filename)
    for i in range(count):
        pda.dump_append(i)
    pda.close()
    input_file = open(filename, 'rb')
    assert load(input_file) == metadata
    for i in range(count):
        data = load(input_file)
        assert data == i
    input_file.close()
    remove_and_check(filename)

def test_PickleDumpAppend_close():
    metadata = {'total': 0}
    filename = generate_available_filename()
    for expected_value in [False, True]:
        assert exists(filename) == expected_value
        pda = pickle_manager.PickleDumpAppend(metadata, filename)
        assert not pda.file.closed
        assert exists(pda.file.name)
        pda.close()
        assert pda.file.closed
        assert not exists(pda.file.name)
    remove_and_check(filename)
