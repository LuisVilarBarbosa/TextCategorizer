import pytest
from text_categorizer.Document import Document
from text_categorizer.document_updater import initial_code_to_run_on_document

def test_initial_code_to_run_on_document():
    index = -1
    fields = dict()
    analyzed_sentences = None
    doc = Document(index=index, fields=fields, analyzed_sentences=analyzed_sentences)
    assert doc.index == index
    assert doc.fields == fields
    assert doc.analyzed_sentences == analyzed_sentences
    initial_code_to_run_on_document(doc)
    assert doc.index == index
    assert doc.fields == fields
    assert doc.analyzed_sentences == analyzed_sentences
