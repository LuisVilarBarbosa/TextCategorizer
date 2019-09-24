#!/usr/bin/python3
# coding=utf-8

from configparser import ConfigParser
from flair.embeddings import DocumentPoolEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from text_categorizer import classifiers

class Parameters:
    def __init__(self, config_filename):
        config = ConfigParser()
        config.read(config_filename)
        self.excel_file = config.get("Preprocessing", "Excel file")
        self.excel_column_with_text_data = config.get("General", "Excel column with text data")
        self.excel_column_with_classification_data = config.get("General", "Excel column with classification data")
        self.nltk_stop_words_package = config.get("Feature extraction", "NLTK stop words package")
        self._load_number_of_jobs(config)
        self.stanfordnlp_language_package = config.get("Preprocessing", "StanfordNLP language package")
        self.stanfordnlp_use_gpu = config.getboolean("Preprocessing", "StanfordNLP use GPU")
        self.stanfordnlp_resources_dir = config.get("Preprocessing", "StanfordNLP resources directory")
        self.preprocessed_data_file = config.get("General", "Preprocessed data file")
        self.preprocessed_data = config.getboolean("Preprocessing", "Preprocess data")
        self.document_adjustment_code = config.get("Feature extraction", "Document adjustment script")
        self._load_vectorizer(config)
        self.use_lda = config.getboolean("Feature extraction", "Use LDA")
        self._load_accepted_probs(config)
        self._load_classifiers(config)
        self.test_subset_size = config.getfloat("Classification", "Test subset size")
        self.force_subsets_regeneration = config.getboolean("Classification", "Force regeneration of training and test subsets")
        self.remove_adjectives = config.getboolean("Feature extraction", "Remove adjectives")
        self._load_synonyms_file(config)
        self._load_resampling(config)
        self.features_file = config.get("Feature extraction", "Features file")
    
    def _load_number_of_jobs(self, config):
        self.number_of_jobs = config.get("General", "Number of jobs")
        if self.number_of_jobs == "None":
            self.number_of_jobs = None
        else:
            self.number_of_jobs = int(self.number_of_jobs)
    
    def _load_vectorizer(self, config):
        self.vectorizer = config.get("Feature extraction", "Vectorizer")
        assert self.vectorizer in [TfidfVectorizer.__name__, CountVectorizer.__name__, HashingVectorizer.__name__, DocumentPoolEmbeddings.__name__]
    
    def _load_accepted_probs(self, config):
        n_accepted_probs = config.get("Classification", "Number of probabilities accepted").split(",")
        self.set_num_accepted_probs = set(map(lambda v: int(v), n_accepted_probs))
        assert len(self.set_num_accepted_probs) > 0
        for v in self.set_num_accepted_probs:
            assert v >= 1
    
    def _load_classifiers(self, config):
        clfs = [
            classifiers.RandomForestClassifier,
            classifiers.BernoulliNB,
            classifiers.MultinomialNB,
            classifiers.ComplementNB,
            classifiers.KNeighborsClassifier,
            classifiers.MLPClassifier,
            classifiers.LinearSVC,
            classifiers.DecisionTreeClassifier,
            classifiers.ExtraTreeClassifier,
            classifiers.DummyClassifier,
            classifiers.SGDClassifier,
            classifiers.BaggingClassifier,
        ]
        clfs_names = config.get("Classification", "Classifiers").split(",")
        self.classifiers = []
        for clf in clfs:
            if clf.__name__ in clfs_names:
                self.classifiers.append(clf)
        assert len(self.classifiers) > 0
    
    def _load_synonyms_file(self, config):
        self.synonyms_file = config.get("Feature extraction", "Synonyms file")
        if self.synonyms_file == "None":
            self.synonyms_file = None
    
    def _load_resampling(self, config):
        self.resampling = config.get("Classification", "Resampling")
        assert self.resampling in ["None", "RandomOverSample", "RandomUnderSample"]
        if self.resampling == "None":
            self.resampling = None
