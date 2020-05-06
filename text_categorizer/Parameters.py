from configparser import ConfigParser
from multiprocessing import cpu_count
from flair.embeddings import DocumentPoolEmbeddings
from os.path import abspath
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from text_categorizer import classifiers

class Parameters:
    def __init__(self, config_filename):
        config = ConfigParser()
        config.read(config_filename)
        self.excel_file = abspath(config.get("Preprocessing", "Excel file"))
        self.excel_column_with_text_data = config.get("General", "Excel column with text data")
        self.excel_column_with_classification_data = config.get("General", "Excel column with classification data")
        self.nltk_stop_words_package = Parameters._parse_None(config.get("Feature extraction", "NLTK stop words package"))
        self.number_of_jobs = Parameters._parse_number_of_jobs(config.get("General", "Number of jobs"))
        self.mosestokenizer_language_code = config.get("Preprocessing", "MosesTokenizer language code")
        self.preprocessed_data_file = abspath(config.get("General", "Preprocessed data file"))
        self.preprocess_data = config.getboolean("Preprocessing", "Preprocess data")
        self.document_adjustment_code = abspath(config.get("Feature extraction", "Document adjustment script"))
        self.vectorizer = Parameters._parse_vectorizer(config.get("Feature extraction", "Vectorizer"))
        self.feature_reduction = Parameters._parse_feature_reduction(config.get("Feature extraction", "Feature reduction"))
        self.set_num_accepted_probs = Parameters._parse_accepted_probs(config.get("Classification", "Number of probabilities accepted"))
        self.classifiers = Parameters._parse_classifiers(config.get("Classification", "Classifiers"))
        self.test_subset_size = config.getfloat("Classification", "Test subset size")
        self.force_subsets_regeneration = config.getboolean("Classification", "Force regeneration of training and test subsets")
        self.remove_adjectives = config.getboolean("Feature extraction", "Remove adjectives")
        self.synonyms_file = Parameters._parse_synonyms_file(config.get("Feature extraction", "Synonyms file"))
        self.resampling = Parameters._parse_resampling(config.get("Classification", "Resampling"))
        self.class_weights = Parameters._parse_class_weights(config.get("Classification", "Class weights"))
        self.generate_roc_plots = config.getboolean("Classification", "Generate ROC plots")
        self.spell_checker_lang = Parameters._parse_None(config.get("Preprocessing", "Spell checker language"))
        self.final_training = config.getboolean("General", "Final training")
        self.data_dir = abspath(config.get("General", "Data directory"))
    
    @staticmethod
    def _parse_number_of_jobs(number_of_jobs):
        my_number_of_jobs = Parameters._parse_None(number_of_jobs)
        if my_number_of_jobs is None:
            my_number_of_jobs = 1
        else:
            my_number_of_jobs = int(my_number_of_jobs)
        if my_number_of_jobs < 0:
            my_number_of_jobs = cpu_count() + 1 + my_number_of_jobs
        assert my_number_of_jobs != 0
        return my_number_of_jobs
    
    @staticmethod
    def _parse_vectorizer(vectorizer):
        accepted_vectorizers = [
            TfidfVectorizer.__name__,
            CountVectorizer.__name__,
            HashingVectorizer.__name__,
            DocumentPoolEmbeddings.__name__,
        ]
        assert vectorizer in accepted_vectorizers
        return vectorizer
    
    @staticmethod
    def _parse_accepted_probs(n_accepted_probs):
        set_num_accepted_probs = set(map(lambda v: int(v), n_accepted_probs.split(",")))
        for v in set_num_accepted_probs:
            assert v >= 1
        return set_num_accepted_probs
    
    @staticmethod
    def _parse_classifiers(clfs):
        accepted_clfs = [
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
        clfs_names = clfs.split(",")
        my_classifiers = []
        for clf in accepted_clfs:
            if clf.__name__ in clfs_names:
                my_classifiers.append(clf)
        assert len(my_classifiers) > 0
        assert len(my_classifiers) == len(clfs_names)
        return my_classifiers
    
    @staticmethod
    def _parse_synonyms_file(synonyms_file):
        my_synonyms_file = Parameters._parse_None(synonyms_file)
        if my_synonyms_file is None:
            return my_synonyms_file
        else:
            return abspath(my_synonyms_file)
    
    @staticmethod
    def _parse_resampling(resampling):
        assert resampling in ["None", "RandomOverSample", "RandomUnderSample"]
        return Parameters._parse_None(resampling)
    
    @staticmethod
    def _parse_class_weights(class_weights):
        assert class_weights in ["None", "balanced"]
        return Parameters._parse_None(class_weights)

    @staticmethod
    def _parse_feature_reduction(feature_reduction):
        assert feature_reduction in ["None", "LDA", "MDS"]
        return Parameters._parse_None(feature_reduction)

    @staticmethod
    def _parse_None(value):
        if value == "None":
            return None
        return value
