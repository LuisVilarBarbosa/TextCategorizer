#!/usr/bin/python3
# coding=utf-8

import numpy as np
from collections import Counter
from flair.embeddings import DocumentPoolEmbeddings, Sentence, BertEmbeddings
from nltk import download
from nltk.corpus import stopwords
from os.path import exists
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.manifold import MDS
from text_categorizer import pickle_manager
from text_categorizer.constants import random_state
from text_categorizer.ContoPTParser import ContoPTParser
from text_categorizer.functions import load_module
from text_categorizer.logger import logger
from text_categorizer.ui import get_documents, progress

class FeatureExtractor:
    def __init__(self, nltk_stop_words_package=None, vectorizer_name="TfidfVectorizer", training_mode=True, feature_reduction=None, document_adjustment_code="text_categorizer/document_updater.py", remove_adjectives=False, synonyms_file=None, vectorizer_file="vectorizer.pkl"):
        download(info_or_id='stopwords', quiet=True)
        self.stop_words = set() if nltk_stop_words_package is None else set(stopwords.words(nltk_stop_words_package))
        self.vectorizer_file = vectorizer_file
        self.vectorizer = FeatureExtractor._get_vectorizer(vectorizer_name, training_mode, vectorizer_file=self.vectorizer_file)
        self.feature_reduction = feature_reduction
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

    def prepare(self, class_field, preprocessed_data_file=None, docs=None, training_mode=True):
        description = "Preparing to create classification"
        if docs is None:
            docs = get_documents(preprocessed_data_file, description=description)
        else:
            docs = progress(iterable=docs, desc=description, unit="doc")
        num_ignored = 0
        idxs_to_remove = []
        corpus = []
        classifications = []
        for doc in docs:
            self.document_adjustment_code.initial_code_to_run_on_document(doc)
            if doc.analyzed_sentences is None:
                num_ignored = num_ignored + 1
                idxs_to_remove.append(doc.index)
            lemmas = FeatureExtractor._filter(doc, self.upostags_to_ignore)
            if self.synonyms is not None:
                lemmas = list(map(lambda l: l if self.synonyms.get(l) is None else self.synonyms.get(l), lemmas))
            lemmas = list(filter(lambda l: l not in self.stop_words, lemmas))
            corpus.append(lemmas)
            classifications.append(doc.fields[class_field])
        corpus_str = [FeatureExtractor._generate_corpus(c) for c in corpus]
        if num_ignored > 0:
            logger.warning("%s document(s) ignored." % num_ignored)
        if training_mode:
            idxs_to_remove.extend(FeatureExtractor._find_incompatible_data_indexes(corpus_str, classifications))
        classifications = np.array(classifications, copy=False).tolist()
        return corpus_str, classifications, idxs_to_remove, corpus

    @staticmethod
    def _filter(doc, upostags_to_ignore):
        lemmas = []
        if doc.analyzed_sentences is not None:
            for sentence in doc.analyzed_sentences:
                lemmas.extend([word['lemma'] for word in sentence if word['upostag'] not in upostags_to_ignore])
        return lemmas

    @staticmethod
    def _generate_corpus(lemmas):
        return ' '.join(lemmas)

    def generate_X_y(self, corpus, classifications, training_mode=True):
        logger.info("Running %s." % self.vectorizer.__class__.__name__)
        if training_mode:
            logger.debug("%s configuration: %s" % (self.vectorizer.__class__.__name__, self.vectorizer.__dict__))
        corpus = progress(iterable=corpus, desc="Extracting features", unit="doc")
        if training_mode and "fit_transform" in dir(self.vectorizer):
            X = self.vectorizer.fit_transform(corpus)
        elif not training_mode and "transform" in dir(self.vectorizer):
            X = self.vectorizer.transform(corpus)
        else:
            X = np.asarray([FeatureExtractor.chunked_embed(t, self.vectorizer) for t in corpus])
        y = classifications
        if training_mode and self.vectorizer.__class__ not in [DocumentPoolEmbeddings]:
            pickle_manager.dump(self.vectorizer, self.vectorizer_file)
        if self.feature_reduction is None:
            return X, y
        elif self.feature_reduction == "LDA":
            return FeatureExtractor.LatentDirichletAllocation(X, y)
        elif self.feature_reduction == "MDS":
            return FeatureExtractor.MDS(X, y)
        else:
            raise ValueError("Invalid feature reduction technique: %s" % (self.feature_reduction))

    @staticmethod
    def _find_incompatible_data_indexes(corpus, classifications):
        logger.info("Searching for incompatible data.")
        quantities = Counter(classifications)
        idxs_to_remove = []
        for k, v in quantities.items():
            if v <= 1:
                logger.warning("Ignoring documents with the classification '%s' because the classification only occurs %d time(s)." % (k, v))
                for _ in range(v):
                    index = classifications.index(k)
                    idxs_to_remove.append(index)
        return idxs_to_remove

    @staticmethod
    def LatentDirichletAllocation(X, y, filename='LatentDirichletAllocation.pkl'):
        logger.info("Running %s." % (LatentDirichletAllocation.__name__))
        if exists(filename):
            lda = pickle_manager.load(filename)
            X = lda.transform(X)
        else:
            lda = LatentDirichletAllocation(n_components=10, doc_topic_prior=None,
                    topic_word_prior=None, learning_method='batch', learning_decay=0.7,
                    learning_offset=10.0, max_iter=10, batch_size=128, evaluate_every=-1,
                    total_samples=1000000.0, perp_tol=0.1, mean_change_tol=0.001,
                    max_doc_update_iter=100, n_jobs=None, verbose=0, random_state=random_state,
                    n_topics=None)
            logger.debug("%s configuration: %s" % (lda.__class__.__name__, lda.__dict__))
            X = lda.fit_transform(X, y)
            pickle_manager.dump(lda, filename)
        return X, y

    @staticmethod
    def _get_vectorizer(vectorizer, training_mode, vectorizer_file="vectorizer.pkl"):
        token_pattern = r'\S+'
        if not training_mode and vectorizer not in [DocumentPoolEmbeddings.__name__]:
                v = pickle_manager.load(vectorizer_file)
                assert vectorizer == v.__class__.__name__
        elif vectorizer == TfidfVectorizer.__name__:
            v = TfidfVectorizer(input='content', encoding='utf-8',
                    decode_error='strict', strip_accents=None, lowercase=True,
                    preprocessor=None, tokenizer=None, analyzer='word',
                    stop_words=[], token_pattern=token_pattern,
                    ngram_range=(1,1), max_df=1.0, min_df=1, max_features=None,
                    vocabulary=None, binary=False, dtype=np.float64, norm='l2',
                    use_idf=True, smooth_idf=True, sublinear_tf=False)
        elif vectorizer == CountVectorizer.__name__:
            v = CountVectorizer(input='content', encoding='utf-8',
                    decode_error='strict', strip_accents=None, lowercase=True,
                    preprocessor=None, tokenizer=None, stop_words=[],
                    token_pattern=token_pattern, ngram_range=(1, 1),
                    analyzer='word', max_df=1.0, min_df=1, max_features=None,
                    vocabulary=None, binary=False, dtype=np.int64)
        elif vectorizer == HashingVectorizer.__name__:
            v = HashingVectorizer(input='content', encoding='utf-8',
                    decode_error='strict', strip_accents=None, lowercase=True,
                    preprocessor=None, tokenizer=None, stop_words=[],
                    token_pattern=token_pattern, ngram_range=(1, 1),
                    analyzer='word', n_features=1048576, binary=False,
                    norm='l2', alternate_sign=True, non_negative=False,
                    dtype=np.float64)
        elif vectorizer == DocumentPoolEmbeddings.__name__:
            v = DocumentPoolEmbeddings([BertEmbeddings('bert-base-multilingual-uncased')])
        else:
            raise "Invalid vectorizer."
        return v

    @staticmethod
    def chunked_embed(corpus, embeddings, chunk_size=256):
        def find_nth(n, substring, text, start):
            index = start
            for _ in range(n):
                index = text.find(substring, index + 1)
            return index
        try:
            partial_embeddings = []
            i = 0
            while i < len(corpus):
                next_i = find_nth(chunk_size, " ", corpus, i)
                if next_i < i:
                    next_i = len(corpus)
                chunk = corpus[i:next_i]
                sentence = Sentence(chunk, use_tokenizer=False)
                embeddings.embed(sentence)
                partial_embeddings.append(sentence.get_embedding().numpy())
                i = next_i
            avg = np.average(np.asarray(partial_embeddings), axis=0)
            return avg
        except RuntimeError:
            print("Please, ignore the message above indicating that the sentence is too long. The problem has been solved.")
            return FeatureExtractor.chunked_embed(corpus, embeddings, int(chunk_size / 2))

    @staticmethod
    def MDS(X, y):
        logger.info("Running %s." % (MDS.__name__))
        mds = MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001,
                    n_jobs=None, random_state=random_state, dissimilarity='euclidean')
        logger.debug("%s configuration: %s" % (mds.__class__.__name__, mds.__dict__))
        if 'toarray' in dir(X):
            X = X.toarray()
        X = mds.fit_transform(X, y)
        return X, y
