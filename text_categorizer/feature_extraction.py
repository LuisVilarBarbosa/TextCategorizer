#!/usr/bin/python3
# coding=utf-8

import numpy as np
import parameters

from sklearn.feature_extraction.text import TfidfVectorizer

def generate_X_y(docs, classification):
    filtered_docs = filter(docs)
    corpus = generate_corpus(filtered_docs)
    X, y = create_classification(corpus, classification)
    return X, y

def filter(docs):
    from nltk import download
    download(info_or_id='stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop_words = set()
    for language in parameters.NLTK_STOP_WORDS_PACKAGES:
        stop_words = stop_words.union(set(stopwords.words(language)))
    filtered_docs = docs.copy()
    for doc in filtered_docs:
        for sentence in doc.sentences:
            new_words = []
            for word in sentence.words:
                if word.upos != 'PUNCT' and not word.lemma in stop_words:  # 'word.lemma' is in lowercase.
                    new_words.append(word)
            sentence.words = new_words
    return filtered_docs

def generate_corpus(docs):
    texts = []
    for doc in docs:
        text = ""
        for sentence in doc.sentences:
            for word in sentence.words:
                text = ' '.join([text, word.lemma])
        texts.append(text)
    return texts

def create_classification(corpus, classification):
    # The code below is based on https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html (accessed on 2019-02-27).
    vectorizer = TfidfVectorizer(input='content', encoding='utf-8',
                decode_error='strict', strip_accents=None, lowercase=True,
                preprocessor=None, tokenizer=None, analyzer='word',
                stop_words=None, token_pattern=r'\S+',
                ngram_range=(1,1), max_df=1.0, min_df=1, max_features=None,
                vocabulary=None, binary=False, dtype=np.float64, norm='l2',
                use_idf=True, smooth_idf=True, sublinear_tf=False)
    X = vectorizer.fit_transform(corpus)
    y = classification
    #print(vectorizer.get_feature_names())
    #print(X.shape)
    return X, y
