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

PREPROCESSED_DATA_FILE = 'data.pkl'

# This boolean indicates whether the preprocessing phase should be performed or not.
# 'True' indicates that the preprocessing phase must be performed and the preprocessed
# data stored in 'PREPROCESSED_DATA_FILE'.
# 'False' indicates that the preprocessing phase should be skipped and the preprocessed
# data loaded from 'PREPROCESSED_DATA_FILE'.
PREPROCESS_DATA = True # TODO: An exception is thrown if 'False' and the file does not exist.

def initial_code_to_run_on_text_data(texts):
    return texts

def initial_code_to_run_on_classification_data(classifications):
    return classifications
