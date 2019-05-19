#!/usr/bin/python3
# coding=utf-8

import importlib.util
import numpy as np
import pickle_manager

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from logger import logger
from Parameters import Parameters
from ui import get_documents

def generate_X_y(docs=None):
    if docs is None:
        docs = get_documents(Parameters.PREPROCESSED_DATA_FILE, description="Preparing to create classification")
    document_adjustment_code = load_document_adjustment_code(Parameters.DOCUMENT_ADJUSTMENT_CODE)
    upostags_to_ignore = ['PUNCT']
    if Parameters.REMOVE_ADJECTIVES:
        upostags_to_ignore.append('ADJ')
    num_ignored = 0
    corpus = []
    classifications = []
    for doc in docs:
        document_adjustment_code.initial_code_to_run_on_document(doc)
        if doc.analyzed_sentences is None:
            num_ignored = num_ignored + 1
        else:
            lemmas = my_filter(doc, upostags_to_ignore)
            corpus.append(generate_corpus(lemmas))
            classifications.append(doc.fields[Parameters.EXCEL_COLUMN_WITH_CLASSIFICATION_DATA])
    if num_ignored > 0:
        logger.warning("%s document(s) ignored." % num_ignored)
    if Parameters.TRAINING_MODE:
        remove_incompatible_data(corpus, classifications)
    X, y = create_classification(corpus=corpus,
                                 classifications=classifications,
                                 nltk_stop_words_package=Parameters.NLTK_STOP_WORDS_PACKAGE,
                                 use_lda=Parameters.USE_LDA,
                                 vectorizer=Parameters.VECTORIZER,
                                 store_vocabulary=Parameters.TRAINING_MODE,
                                 training_mode=Parameters.TRAINING_MODE)
    return X, y, lemmas

def load_document_adjustment_code(filename):
    spec = importlib.util.spec_from_file_location("document_adjustment_code", filename)
    document_adjustment_code = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(document_adjustment_code)
    return document_adjustment_code

def my_filter(doc, upostags_to_ignore):
    lemmas = []
    for sentence in doc.analyzed_sentences:
        lemmas.extend([word['lemma'] for word in sentence if word['upostag'] not in upostags_to_ignore])
    return lemmas

def generate_corpus(lemmas):
    return ' '.join(lemmas)

def create_classification(corpus, classifications, nltk_stop_words_package, use_lda, vectorizer, store_vocabulary, training_mode):
    logger.info("Creating classification.")
    from nltk import download
    download(info_or_id='stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words(nltk_stop_words_package))
    vectorizer = get_vectorizer(vectorizer, stop_words=stop_words, check_vectorizer=False, training_mode=training_mode)
    logger.info("Running %s." % vectorizer.__class__.__name__)
    logger.debug("%s configuration: %s" % (vectorizer.__class__.__name__, vectorizer.__dict__))
    X = vectorizer.fit_transform(corpus)
    y = classifications
    if store_vocabulary and vectorizer.__class__ != HashingVectorizer:
        pickle_manager.dump(vectorizer.vocabulary_, "features.pkl")
    if use_lda:
        X, y = LatentDirichletAllocation(X, y)
    #logger.debug(vectorizer.get_feature_names())
    #logger.debug(X.shape)
    return X, y

def remove_incompatible_data(corpus, classifications):
    from collections import Counter
    logger.info("Removing incompatible data.")
    quantities = Counter(classifications)
    for k, v in quantities.items():
        if v <= 1:
            logger.warning("Ignoring documents with the classification '%s' because the classification only occurs %d time(s)." % (k, v))
            for _ in range(v):
                index = classifications.index(k)
                corpus.pop(index)
                classifications.pop(index)

def LatentDirichletAllocation(X, y):
    from sklearn.decomposition import LatentDirichletAllocation
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

def get_vectorizer(vectorizer, stop_words=[], check_vectorizer=False, training_mode=True): #TODO: 'training_mode' should not have a default value.
    if check_vectorizer:
        assert vectorizer in [TfidfVectorizer.__name__, CountVectorizer.__name__, HashingVectorizer.__name__]
        return
    token_pattern = r'\S+'
    if training_mode:
        vocabulary = None
    else:
        if vectorizer != HashingVectorizer.__name__:
            vocabulary = pickle_manager.load("features.pkl")
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
