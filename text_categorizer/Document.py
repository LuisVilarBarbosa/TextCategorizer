#!/usr/bin/python3
# coding=utf-8

import conllu
import pandas
import stanfordnlp

_excel_start_row = 2

class Document:
    def __init__(self, index, fields, analyzed_sentences):
        self.index = index
        self.fields = fields
        self.analyzed_sentences = analyzed_sentences

    @staticmethod
    def from_data_frame(data_frame, index):
        assert type(data_frame) is pandas.DataFrame
        assert type(index) is int
        doc_index = index + _excel_start_row
        fields = dict()
        columns = data_frame.columns
        for i in range(len(columns)):
            fields[columns[i]] = data_frame.iloc[index, i]
        return Document(index=doc_index, fields=fields, analyzed_sentences=None)

    def update(self, stanfordnlp_document, text_data_field):
        assert type(stanfordnlp_document) is stanfordnlp.Document
        assert self.fields[text_data_field] == stanfordnlp_document.text
        if stanfordnlp_document.conll_file is not None:
            conll = stanfordnlp_document.conll_file.conll_as_string()
            self.analyzed_sentences = conllu.parse(conll)

    def copy(self):
        if self.analyzed_sentences is None:
            analyzed_sentences = None
        else:
            analyzed_sentences = self.analyzed_sentences.copy()
        return Document(self.index, self.fields.copy(), analyzed_sentences)

    def __repr__(self):
        return "%s: %s" % (self.__class__.__name__, self.__dict__)
