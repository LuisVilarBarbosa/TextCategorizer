#!/usr/bin/python3
# coding=utf-8

class Parameters:
    EXCEL_FILE = None
    EXCEL_COLUMN_WITH_TEXT_DATA = None
    EXCEL_COLUMN_WITH_CLASSIFICATION_DATA = None
    NLTK_STOP_WORDS_PACKAGE = None
    NUMBER_OF_PROCESSES = None
    STANFORDNLP_LANGUAGE_PACKAGE = None
    STANFORDNLP_USE_GPU = None
    STANFORDNLP_RESOURCES_DIR = None
    PREPROCESSED_DATA_FILE = None
    PREPROCESS_DATA = None
    EXCEL_FILTRATION_CODE = None

    # This function must be executed before any access to the static variables of the class.
    @staticmethod
    def load_configuration(config_filename):
        from configparser import ConfigParser
        config = ConfigParser()
        config.read(config_filename)
        Parameters.EXCEL_FILE = config.get("Preprocessing", "Excel file")
        Parameters.EXCEL_COLUMN_WITH_TEXT_DATA = config.get("Preprocessing", "Excel column with text data")
        Parameters.EXCEL_COLUMN_WITH_CLASSIFICATION_DATA = config.get("Preprocessing", "Excel column with classification data")
        Parameters.NLTK_STOP_WORDS_PACKAGE = config.get("Feature extraction", "NLTK stop words package")
        Parameters.STANFORDNLP_LANGUAGE_PACKAGE = config.get("Preprocessing", "StanfordNLP language package")
        Parameters.STANFORDNLP_USE_GPU = config.get("Preprocessing", "StanfordNLP use GPU")
        Parameters.STANFORDNLP_RESOURCES_DIR = config.get("Preprocessing", "StanfordNLP resources directory")
        Parameters.PREPROCESSED_DATA_FILE = config.get("Preprocessing", "Preprocessed data file")
        Parameters.PREPROCESS_DATA = config.getboolean("Preprocessing", "Preprocess data")
        Parameters.EXCEL_FILTRATION_CODE = config.get("Preprocessing", "Excel filtration script")
