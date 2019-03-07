#!/usr/bin/python3
# coding=utf-8

import conllu
import pandas
import stanfordnlp
import parameters

class Document:
    def __init__(self, data_frame, index, stanfordnlp_document=None):
        assert type(data_frame) is pandas.DataFrame
        assert type(index) is int
        assert type(stanfordnlp_document) in [stanfordnlp.Document, type(None)]
        self.fields = dict()
        columns = data_frame.columns
        for i in range(len(columns)):
            self.fields[columns[i]] = data_frame.iloc[index, i]
        self.update_stanfordnlp_document(stanfordnlp_document)

    def update_stanfordnlp_document(self, stanfordnlp_document):
        self.analyzed_sentences = None
        if stanfordnlp_document is not None:
            assert self.fields[parameters.EXCEL_COLUMN_WITH_TEXT_DATA] == stanfordnlp_document.text
            if stanfordnlp_document.conll_file is not None:
                conll = stanfordnlp_document.conll_file.conll_as_string()
                self.analyzed_sentences = conllu.parse(conll)
