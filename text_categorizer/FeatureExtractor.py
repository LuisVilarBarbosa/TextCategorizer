#!/usr/bin/python3
# coding=utf-8

import numpy as np
import pickle_manager

from collections import Counter
from nltk import download
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from ContoPTParser import ContoPTParser
from functions import load_module
from logger import logger
from ui import get_documents

class FeatureExtractor:
    def __init__(self, nltk_stop_words_package="english", vectorizer_name="TfidfVectorizer", training_mode=True, use_lda=False, document_adjustment_code="text_categorizer/document_updater.py", remove_adjectives=False, synonyms_file=None, features_file="features.pkl"):
        download(info_or_id='stopwords', quiet=True)
        self.stop_words = set(stopwords.words(nltk_stop_words_package))
        self.training_mode = training_mode
        self.features_file = features_file
        self.vectorizer = FeatureExtractor._get_vectorizer(vectorizer_name, self.training_mode, stop_words=self.stop_words, features_file=self.features_file)
        self.use_lda = use_lda
        self.document_adjustment_code = load_module(document_adjustment_code)
        self.upostags_to_ignore = ['PUNCT']
        if remove_adjectives:
            logger.info("The removal of adjectives is enabled.")
            self.upostags_to_ignore.append('ADJ')
        else:
            logger.info("The removal of adjectives is disabled.")
        if synonyms_file is None:
            logger.info("The substitution of synonyms is disabled.")
            self.synonyms = None
        else:
            logger.info("The substitution of synonyms is enabled.")
            contoPTParser = ContoPTParser(synonyms_file)
            self.synonyms = contoPTParser.synonyms

    def generate_X_y(self, class_field, preprocessed_data_file=None, docs=None):
        if docs is None:
            docs = get_documents(preprocessed_data_file, description="Preparing to create classification")
        num_ignored = 0
        corpus = []
        classifications = []
        for doc in docs:
            self.document_adjustment_code.initial_code_to_run_on_document(doc)
            if doc.analyzed_sentences is None:
                num_ignored = num_ignored + 1
            else:
                lemmas = FeatureExtractor._filter(doc, self.upostags_to_ignore)
                if self.synonyms is not None:
                    lemmas = list(map(lambda l: l if self.synonyms.get(l) is None else self.synonyms.get(l), lemmas))
                corpus.append(FeatureExtractor._generate_corpus(lemmas))
                classifications.append(doc.fields[class_field])
        if num_ignored > 0:
            logger.warning("%s document(s) ignored." % num_ignored)
        if self.training_mode:
            FeatureExtractor._remove_incompatible_data(corpus, classifications)
        X, y = self._create_classification(corpus, classifications)
        return X, y, lemmas

    @staticmethod
    def _filter(doc, upostags_to_ignore):
        lemmas = []
        for sentence in doc.analyzed_sentences:
            lemmas.extend([word['lemma'] for word in sentence if word['upostag'] not in upostags_to_ignore])
        return lemmas

    @staticmethod
    def _generate_corpus(lemmas):
        return ' '.join(lemmas)

    def _create_classification(self, corpus, classifications):
        logger.info("Creating classification.")
        logger.info("Running %s." % self.vectorizer.__class__.__name__)
        if self.training_mode:
            logger.debug("%s configuration: %s" % (self.vectorizer.__class__.__name__, self.vectorizer.__dict__))
        X = self.vectorizer.fit_transform(corpus)
        y = classifications
        if self.training_mode and self.vectorizer.__class__ != HashingVectorizer:
            pickle_manager.dump(self.vectorizer.vocabulary_, self.features_file)
        if self.use_lda:
            X, y = FeatureExtractor._LatentDirichletAllocation(X, y)
        #logger.debug(self.vectorizer.get_feature_names())
        #logger.debug(X.shape)
        return X, y

    @staticmethod
    def _remove_incompatible_data(corpus, classifications):
        logger.info("Removing incompatible data.")
        quantities = Counter(classifications)
        for k, v in quantities.items():
            if v <= 1:
                logger.warning("Ignoring documents with the classification '%s' because the classification only occurs %d time(s)." % (k, v))
                for _ in range(v):
                    index = classifications.index(k)
                    corpus.pop(index)
                    classifications.pop(index)

    @staticmethod
    def _LatentDirichletAllocation(X, y):
        logger.info("Running %s." % (LatentDirichletAllocation.__name__))
        lda = LatentDirichletAllocation(n_components=10, doc_topic_prior=None,
                    topic_word_prior=None, learning_method='batch', learning_decay=0.7,
                    learning_offset=10.0, max_iter=10, batch_size=128, evaluate_every=-1,
                    total_samples=1000000.0, perp_tol=0.1, mean_change_tol=0.001,
                    max_doc_update_iter=100, n_jobs=None, verbose=0, random_state=None,
                    n_topics=None)
        logger.debug("%s configuration: %s" % (lda.__class__.__name__, lda.__dict__))
        X = lda.fit_transform(X, y)
        return X, y

    @staticmethod
    def _get_vectorizer(vectorizer, training_mode, stop_words=[], features_file="features.pkl"):
        token_pattern = r'\S+'
        if training_mode:
            vocabulary = None
        else:
            if vectorizer != HashingVectorizer.__name__:
                vocabulary = pickle_manager.load(features_file)
        if vectorizer == TfidfVectorizer.__name__:
            v = TfidfVectorizer(input='content', encoding='utf-8',
                    decode_error='strict', strip_accents=None, lowercase=True,
                    preprocessor=None, tokenizer=None, analyzer='word',
                    stop_words=stop_words, token_pattern=token_pattern,
                    ngram_range=(1,1), max_df=1.0, min_df=1, max_features=None,
                    vocabulary=vocabulary, binary=False, dtype=np.float64, norm='l2',
                    use_idf=True, smooth_idf=True, sublinear_tf=False)
        elif vectorizer == CountVectorizer.__name__:
            v = CountVectorizer(input='content', encoding='utf-8',
                    decode_error='strict', strip_accents=None, lowercase=True,
                    preprocessor=None, tokenizer=None, stop_words=stop_words,
                    token_pattern=token_pattern, ngram_range=(1, 1),
                    analyzer='word', max_df=1.0, min_df=1, max_features=None,
                    vocabulary=vocabulary, binary=False, dtype=np.int64)
        elif vectorizer == HashingVectorizer.__name__:
            v = HashingVectorizer(input='content', encoding='utf-8',
                    decode_error='strict', strip_accents=None, lowercase=True,
                    preprocessor=None, tokenizer=None, stop_words=stop_words,
                    token_pattern=token_pattern, ngram_range=(1, 1),
                    analyzer='word', n_features=1048576, binary=False,
                    norm='l2', alternate_sign=True, non_negative=False,
                    dtype=np.float64)
        else:
            raise "Invalid vectorizer."
        return v
