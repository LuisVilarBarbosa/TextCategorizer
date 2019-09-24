from itertools import zip_longest
from pandas import read_excel
from text_categorizer.functions import data_frame_to_document_list
from text_categorizer.pickle_manager import dump_documents
from text_categorizer.ui import get_documents, verify_python_version
from tests.utils import create_temporary_file, example_excel_file, generate_available_filename, remove_and_check

def test_verify_python_version():
    verify_python_version()

def test_get_documents():
    df = read_excel(example_excel_file)
    docs1 = data_frame_to_document_list(df)
    filename = generate_available_filename()
    dump_documents(docs1, filename)
    docs2 = get_documents(filename)
    for doc1, doc2 in zip_longest(docs1, docs2):
        assert repr(doc1) == repr(doc2)
    remove_and_check(filename)
