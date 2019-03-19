#!/usr/bin/python3
# coding=utf-8

import numpy as np
import pickle_manager

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from Parameters import Parameters

def generate_X_y():
    docs = pickle_manager.get_documents()
    num_ignored = 0
    corpus = []
    classifications = []
    for doc in tqdm(iterable=docs, desc="Preparing to create classification", total=pickle_manager.get_total_docs(), unit="doc"):
        if doc.analyzed_sentences is None:
            num_ignored = num_ignored + 1
        else:
            lemmas = my_filter(doc)
            corpus.append(generate_corpus(lemmas))
            classifications.append(get_classification(doc))
    print("Warning - %s document(s) ignored." % num_ignored)
    print("Removing incompatible data.")
    remove_incompatible_data(corpus, classifications)
    print("Creating classification.")
    X, y = create_classification(corpus, classifications)
    return X, y

def my_filter(doc):
    lemmas = []
    for sentence in doc.analyzed_sentences:
        lemmas.extend([word['lemma'] for word in sentence if word['upostag'] != 'PUNCT'])
    return lemmas

def generate_corpus(lemmas):
    return ' '.join(lemmas)

def get_classification(doc):
    return doc.fields[Parameters.EXCEL_COLUMN_WITH_CLASSIFICATION_DATA]

def create_classification(corpus, classifications):
    from nltk import download
    download(info_or_id='stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words(Parameters.NLTK_STOP_WORDS_PACKAGE))
    # The code below is based on https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html (accessed on 2019-02-27).
    vectorizer = TfidfVectorizer(input='content', encoding='utf-8',
                decode_error='strict', strip_accents=None, lowercase=True,
                preprocessor=None, tokenizer=None, analyzer='word',
                stop_words=stop_words, token_pattern=r'\S+',
                ngram_range=(1,1), max_df=1.0, min_df=1, max_features=None,
                vocabulary=None, binary=False, dtype=np.float64, norm='l2',
                use_idf=True, smooth_idf=True, sublinear_tf=False)
    X = vectorizer.fit_transform(corpus)
    y = classifications
    #print(vectorizer.get_feature_names())
    #print(X.shape)
    return X, y

def remove_incompatible_data(corpus, classifications):
    from collections import Counter
    quantities = Counter(classifications)
    for k, v in quantities.items():
        if v <= 1:
            print("Warning - Ignoring documents with the classification '%s' because the classification only occurs %d time(s)." % (k, v))
            for _ in range(v):
                index = classifications.index(k)
                corpus.pop(index)
                classifications.pop(index)

def LatentDirichletAllocation(X, y):
    from sklearn.decomposition import LatentDirichletAllocation
    lda = LatentDirichletAllocation(n_components=10, doc_topic_prior=None,
                topic_word_prior=None, learning_method='batch', learning_decay=0.7,
                learning_offset=10.0, max_iter=10, batch_size=128, evaluate_every=-1,
                total_samples=1000000.0, perp_tol=0.1, mean_change_tol=0.001,
                max_doc_update_iter=100, n_jobs=None, verbose=0, random_state=None,
                n_topics=None)
    X = lda.fit_transform(X, y)
    return X, y
