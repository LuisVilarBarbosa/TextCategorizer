#!/usr/bin/python3
# coding=utf-8

import numpy as np
import parameters
import pickle_manager

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

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
    return doc.fields[parameters.EXCEL_COLUMN_WITH_CLASSIFICATION_DATA]

def create_classification(corpus, classifications):
    from nltk import download
    download(info_or_id='stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop_words = set()
    for language in parameters.NLTK_STOP_WORDS_PACKAGES:
        stop_words = stop_words.union(set(stopwords.words(language)))
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
