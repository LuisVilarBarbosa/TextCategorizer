#!/usr/bin/python3
# coding=utf-8

# The text present in the first row of the column with the textual data of interest.
EXCEL_COLUMN_WITH_TEXT_DATA = "Example column"

# The text present in the first row of the column with the classifications.
EXCEL_COLUMN_WITH_CLASSIFICATION_DATA = "Classification column"

NLTK_STOP_WORDS_PACKAGES = ['english']

STANFORDNLP_LANGUAGE_PACKAGE = 'en'

STANFORDNLP_USE_GPU = False

STANFORDNLP_RESOURCES_DIR = './stanfordnlp_resources'

def initial_code_to_run_on_text_data(texts):
    return texts

def initial_code_to_run_on_classification_data(classifications):
    return classifications
