#!/usr/bin/python3
# coding=utf-8

# Only used if 'PREPROCESS_DATA' is True.
EXCEL_FILE = "example_excel_file.xlsx"

# The text present in the first row of the column with the textual data of interest.
EXCEL_COLUMN_WITH_TEXT_DATA = "Example column"

# The text present in the first row of the column with the classifications.
EXCEL_COLUMN_WITH_CLASSIFICATION_DATA = "Classification column"

NLTK_STOP_WORDS_PACKAGES = ['english']

# Indicate 0 to use all machine cores.
NUMBER_OF_PROCESSES = 0

STANFORDNLP_LANGUAGE_PACKAGE = 'en'

STANFORDNLP_USE_GPU = False

STANFORDNLP_RESOURCES_DIR = './stanfordnlp_resources'

PREPROCESSED_DATA_FOLDER = './preprocessed_data'

# This boolean indicates whether the preprocessing phase should be performed or not.
# 'True' indicates that the preprocessing phase must be performed and the preprocessed
# data stored in 'PREPROCESSED_DATA_FILE'.
# 'False' indicates that the preprocessing phase should be skipped and the preprocessed
# data loaded from 'PREPROCESSED_DATA_FILE'.
PREPROCESS_DATA = True

def initial_code_to_run_on_data_frame(data_frame):
    return data_frame
