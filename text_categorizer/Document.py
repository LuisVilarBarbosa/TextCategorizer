#!/usr/bin/python3
# coding=utf-8

import conllu
import pandas
import stanfordnlp
import parameters

_excel_start_row = 2

class Document:
    def __init__(self, index, fields, analyzed_sentences):
        self.index = index
        self.fields = fields
        self.analyzed_sentences = analyzed_sentences

    @staticmethod
    def from_data_frame(data_frame, index, stanfordnlp_document=None):
        assert type(data_frame) is pandas.DataFrame
        assert type(index) is int
        assert type(stanfordnlp_document) in [stanfordnlp.Document, type(None)]
        doc_index = index + _excel_start_row
        fields = dict()
        columns = data_frame.columns
        for i in range(len(columns)):
            fields[columns[i]] = data_frame.iloc[index, i]
        doc = Document(index=doc_index, fields=fields, analyzed_sentences=None)
        doc.update_stanfordnlp_document(stanfordnlp_document)
        return doc

    def update_stanfordnlp_document(self, stanfordnlp_document):
        self.analyzed_sentences = None
        if stanfordnlp_document is not None:
            assert self.fields[parameters.EXCEL_COLUMN_WITH_TEXT_DATA] == stanfordnlp_document.text
            if stanfordnlp_document.conll_file is not None:
                conll = stanfordnlp_document.conll_file.conll_as_string()
                self.analyzed_sentences = conllu.parse(conll)

    def copy(self):
        if self.analyzed_sentences is None:
            analyzed_sentences = None
        else:
            analyzed_sentences = self.analyzed_sentences.copy()
        return Document(self.index, self.fields.copy(), analyzed_sentences)
