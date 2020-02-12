from configparser import ConfigParser
from multiprocessing import cpu_count
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
        self._load_nltk_stop_words_package(config)
        self._load_number_of_jobs(config)
        self.stanfordnlp_language_package = config.get("Preprocessing", "StanfordNLP language package")
        self.stanfordnlp_use_gpu = config.getboolean("Preprocessing", "StanfordNLP use GPU")
        self.stanfordnlp_resources_dir = config.get("Preprocessing", "StanfordNLP resources directory")
        self.preprocessed_data_file = config.get("General", "Preprocessed data file")
        self.preprocess_data = config.getboolean("Preprocessing", "Preprocess data")
        self.document_adjustment_code = config.get("Feature extraction", "Document adjustment script")
        self._load_vectorizer(config)
        self._load_feature_reduction(config)
        self._load_accepted_probs(config)
        self._load_classifiers(config)
        self.test_subset_size = config.getfloat("Classification", "Test subset size")
        self.force_subsets_regeneration = config.getboolean("Classification", "Force regeneration of training and test subsets")
        self.remove_adjectives = config.getboolean("Feature extraction", "Remove adjectives")
        self._load_synonyms_file(config)
        self._load_resampling(config)
        self.vectorizer_file = config.get("Feature extraction", "Vectorizer file")
        self._load_class_weights(config)
        self.generate_roc_plots = config.getboolean("Classification", "Generate ROC plots")
        self._load_spell_checker_lang(config)
    
    def _load_number_of_jobs(self, config):
        self.number_of_jobs = config.get("General", "Number of jobs")
        if self.number_of_jobs == "None":
            self.number_of_jobs = 1
        else:
            self.number_of_jobs = int(self.number_of_jobs)
        if self.number_of_jobs < 0:
            self.number_of_jobs = cpu_count() + 1 + self.number_of_jobs
        assert self.number_of_jobs != 0
    
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
        assert len(self.classifiers) == len(clfs_names)
    
    def _load_synonyms_file(self, config):
        self.synonyms_file = config.get("Feature extraction", "Synonyms file")
        if self.synonyms_file == "None":
            self.synonyms_file = None
    
    def _load_resampling(self, config):
        self.resampling = config.get("Classification", "Resampling")
        assert self.resampling in ["None", "RandomOverSample", "RandomUnderSample"]
        if self.resampling == "None":
            self.resampling = None
    
    def _load_class_weights(self, config):
        self.class_weights = config.get("Classification", "Class weights")
        assert self.class_weights in ["None", "balanced"]
        if self.class_weights == "None":
            self.class_weights = None

    def _load_feature_reduction(self, config):
        self.feature_reduction = config.get("Feature extraction", "Feature reduction")
        assert self.feature_reduction in ["None", "LDA", "MDS"]
        if self.feature_reduction == "None":
            self.feature_reduction = None

    def _load_nltk_stop_words_package(self, config):
        self.nltk_stop_words_package = config.get("Feature extraction", "NLTK stop words package")
        if self.nltk_stop_words_package == "None":
            self.nltk_stop_words_package = None

    def _load_spell_checker_lang(self, config):
        self.spell_checker_lang = config.get("Preprocessing", "Spell checker language")
        if self.spell_checker_lang == "None":
            self.spell_checker_lang = None
