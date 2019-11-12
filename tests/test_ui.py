import pytest
from itertools import zip_longest
from pandas import read_excel
from tests.utils import example_excel_file, generate_available_filename, remove_and_check
from text_categorizer.functions import data_frame_to_document_list
from text_categorizer.pickle_manager import dump_documents
from text_categorizer.ui import get_documents, verify_python_version

def test_verify_python_version(monkeypatch):
    verify_python_version()
    with monkeypatch.context() as m:
        m.setattr("text_categorizer.functions.get_python_version", lambda: [3,5,0])
        verify_python_version()
        with pytest.raises(SystemExit):
            m.setattr("text_categorizer.functions.get_python_version", lambda: [3,4,9])
            verify_python_version()

def test_get_documents():
    df = read_excel(example_excel_file)
    docs1 = data_frame_to_document_list(df)
    filename = generate_available_filename()
    try:
        dump_documents(docs1, filename)
        docs2 = list(get_documents(filename))
        for doc1, doc2 in zip_longest(docs1, docs2):
            assert repr(doc1) == repr(doc2)
    finally:
        remove_and_check(filename)
