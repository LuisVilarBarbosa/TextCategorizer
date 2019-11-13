import pytest
from itertools import zip_longest
from pandas import read_excel
from tests.utils import example_excel_file, generate_available_filename, remove_and_check
from text_categorizer.functions import data_frame_to_document_list
from text_categorizer.pickle_manager import dump_documents
from text_categorizer.ui import get_documents, progress, verify_python_version
from tqdm import tqdm

def test_verify_python_version(monkeypatch):
    verify_python_version()
    with monkeypatch.context() as m:
        m.setattr("text_categorizer.functions.get_python_version", lambda: [3,5,0])
        verify_python_version()
        with pytest.raises(SystemExit):
            m.setattr("text_categorizer.functions.get_python_version", lambda: [3,4,9])
            verify_python_version()

def test_get_documents(capsys):
    df = read_excel(example_excel_file)
    docs1 = data_frame_to_document_list(df)
    filename = generate_available_filename()
    try:
        dump_documents(docs1, filename)
        for d1, d2 in [(None, '100%|'), ('Loading documents', 'Loading documents: 100%|')]:
            docs2 = list(get_documents(filename, description=d1))
            for doc1, doc2 in zip_longest(docs1, docs2):
                assert repr(doc1) == repr(doc2)
            captured = capsys.readouterr()
            assert captured.out == ''
            assert captured.err[captured.err.rfind('\r')+1:].startswith(d2)
            assert captured.err.endswith('doc/s]\n') or captured.err.endswith('s/doc]\n')
    finally:
        remove_and_check(filename)

def test_progress():
    assert repr(progress()) == '0it [00:00, ?it/s]'
    assert repr(progress(desc='Test description')) == 'Test description: 0it [00:00, ?it/s]'
    assert repr(progress(unit='doc')) == '0doc [00:00, ?doc/s]'
    assert repr(progress(desc='Test description', unit='doc')) == 'Test description: 0doc [00:00, ?doc/s]'
