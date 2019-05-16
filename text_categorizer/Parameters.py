#!/usr/bin/python3
# coding=utf-8

import classifiers

class Parameters:
    EXCEL_FILE = None
    EXCEL_COLUMN_WITH_TEXT_DATA = None
    EXCEL_COLUMN_WITH_CLASSIFICATION_DATA = None
    NLTK_STOP_WORDS_PACKAGE = None
    NUMBER_OF_JOBS = None
    STANFORDNLP_LANGUAGE_PACKAGE = None
    STANFORDNLP_USE_GPU = None
    STANFORDNLP_RESOURCES_DIR = None
    PREPROCESSED_DATA_FILE = None
    PREPROCESS_DATA = None
    EXCEL_FILTRATION_CODE = None
    CROSS_VALIDATE = None
    VECTORIZER = None
    TRAINING_MODE = None
    USE_LDA = None
    CLASSIFIERS = None
    TEST_SIZE = None

    # This function must be executed before any access to the static variables of the class.
    @staticmethod
    def load_configuration(config_filename, training_mode):
        from configparser import ConfigParser
        from feature_extraction import get_vectorizer
        config = ConfigParser()
        config.read(config_filename)
        Parameters.EXCEL_FILE = config.get("Preprocessing", "Excel file")
        Parameters.EXCEL_COLUMN_WITH_TEXT_DATA = config.get("General", "Excel column with text data")
        Parameters.EXCEL_COLUMN_WITH_CLASSIFICATION_DATA = config.get("General", "Excel column with classification data")
        Parameters.NLTK_STOP_WORDS_PACKAGE = config.get("Feature extraction", "NLTK stop words package")
        Parameters.NUMBER_OF_JOBS = config.get("General", "Number of jobs")
        if Parameters.NUMBER_OF_JOBS == "None":
            Parameters.NUMBER_OF_JOBS = None
        else:
            Parameters.NUMBER_OF_JOBS = int(Parameters.NUMBER_OF_JOBS)
        Parameters.STANFORDNLP_LANGUAGE_PACKAGE = config.get("Preprocessing", "StanfordNLP language package")
        Parameters.STANFORDNLP_USE_GPU = config.getboolean("Preprocessing", "StanfordNLP use GPU")
        Parameters.STANFORDNLP_RESOURCES_DIR = config.get("Preprocessing", "StanfordNLP resources directory")
        Parameters.PREPROCESSED_DATA_FILE = config.get("General", "Preprocessed data file")
        Parameters.PREPROCESS_DATA = config.getboolean("Preprocessing", "Preprocess data")
        Parameters.EXCEL_FILTRATION_CODE = config.get("Preprocessing", "Excel filtration script")
        Parameters.CROSS_VALIDATE = config.getboolean("Classification", "Cross validate")
        Parameters.VECTORIZER = config.get("Feature extraction", "Vectorizer")
        get_vectorizer(Parameters.VECTORIZER, check_vectorizer=True)
        assert type(training_mode) is bool
        Parameters.TRAINING_MODE = training_mode
        Parameters.USE_LDA = config.getboolean("Feature extraction", "Use LDA")
        Parameters.NUM_ACCEPTED_PROBS = config.getint("Classification", "Number of probabilities accepted")
        assert Parameters.NUM_ACCEPTED_PROBS >= 1
        Parameters.load_classifiers(config)
        Parameters.TEST_SIZE = config.getfloat("Classification", "Test subset size")
    
    @staticmethod
    def load_classifiers(config):
        clfs = [
            classifiers.RandomForestClassifier,
            classifiers.BernoulliNB,
            classifiers.MultinomialNB,
            classifiers.ComplementNB,
            classifiers.KNeighborsClassifier,
            classifiers.MLPClassifier,
            classifiers.SVC,
            classifiers.DecisionTreeClassifier,
            classifiers.ExtraTreeClassifier,
            classifiers.DummyClassifier,
            classifiers.SGDClassifier,
            classifiers.BaggingClassifier,
        ]
        clfs_names = config.get("Classification", "Classifiers").split(",")
        Parameters.CLASSIFIERS = []
        for clf in clfs:
            if clf.__name__ in clfs_names:
                Parameters.CLASSIFIERS.append(clf)
