import pytest
from pandas import DataFrame, read_excel
from stanfordnlp.pipeline.doc import Document as SNLPDoc
from text_categorizer.Document import Document, _excel_start_row
from tests.utils import example_excel_file

def test__excel_start_row():
    assert _excel_start_row == 2

def test___init__():
    index = -1
    fields = dict()
    analyzed_sentences = None
    doc = Document(index, fields, analyzed_sentences)
    assert doc.index == index
    assert doc.fields == fields
    assert doc.analyzed_sentences == analyzed_sentences

def test_from_data_frame():
    not_data_frame = 'test_str'
    not_int = 'test_str'
    params = [[not_data_frame, -1], [DataFrame(), not_int]]
    for df, index in params:
        with pytest.raises(AssertionError):
            Document.from_data_frame(df, index)
    df = read_excel(example_excel_file)
    index = 0
    doc = Document.from_data_frame(df, index)
    assert doc.index == index + _excel_start_row
    assert sorted(doc.fields.keys()) == sorted(df.columns)
    assert str(doc.fields) == str(df.to_dict('records')[index])
    assert doc.analyzed_sentences is None

def test_update():
    test_field = 'test_field'
    not_SNLPDoc = 'test_str'
    doc1 = Document(index=-1, fields={test_field: 0}, analyzed_sentences=None)
    params1 = [[not_SNLPDoc, test_field], [SNLPDoc(''), test_field]]
    for snlpdoc, text_data_field in params1:
        with pytest.raises(AssertionError):
            doc1.update(snlpdoc, text_data_field)
    params2 = [[SNLPDoc(''), True], [SNLPDoc('test_text'), True]] #TODO: Add pre-processed SNLP document.
    for snlpdoc, expected_value in params2:
        doc2 = Document(index=-1, fields={test_field: snlpdoc.text}, analyzed_sentences=None)
        doc2.update(snlpdoc, test_field)
        assert (doc2.analyzed_sentences is None) == expected_value

def test_copy():
    test_dict = {'test_field': 'test_value'}
    for analyzed_sentences in [None, test_dict]:
        doc1 = Document(index=-1, fields=test_dict, analyzed_sentences=analyzed_sentences)
        doc2 = doc1.copy()
        assert doc1 is not doc2
        assert doc1.__dict__ == doc2.__dict__
        vars1 = doc1.__dict__
        vars2 = doc2.__dict__
        assert vars1 == vars2
        assert all([var is not int or doc1.__dict__[var] is not doc2.__dict__[var] for var in vars1])

def test___repr__():
    doc = Document(index=-1, fields=[], analyzed_sentences=None)
    assert repr(doc) == "Document: {'index': -1, 'fields': [], 'analyzed_sentences': None}"
