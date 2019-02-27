#!/usr/bin/python3
# coding=utf-8

import parameters

def preprocess(text_array_1d):
    tokenized_array = tokenize(text_array_1d)
    filtered_array = filter(tokenized_array)
    lemmatized_array = lemmatize(filtered_array)
    stemmed_array = stem(lemmatized_array)
    return stemmed_array

def tokenize(text_array_1d):
    from nltk.tokenize import MWETokenizer
    tokenizer = MWETokenizer([])
    tokenized_array = []
    for text_elem in text_array_1d:
        tokenized_array.append(tokenizer.tokenize(text_elem.split()))
    return tokenized_array

def filter(text_array_2d):
    from nltk import download
    download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop_words = set()
    for language in parameters.NLTK_STOP_WORDS_PACKAGES:
        stop_words = stop_words.union(set(stopwords.words(language)))
    # 'stop_words' contains text in lower case. That's not the case in 'text_array_2d'.
    filtered_array = []
    for text_array_1d in text_array_2d:
        new_text_array_1d = []
        for word in text_array_1d:
            if not word in stop_words:
                new_text_array_1d.append(word)
        filtered_array.append(new_text_array_1d)
    return filtered_array

def lemmatize(text_array_2d):
    from nltk import download
    download('wordnet', quiet=True)
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_array = []
    for text_array_1d in text_array_2d:
        new_text_array_1d = []
        for word in text_array_1d:
            new_text_array_1d.append(lemmatizer.lemmatize(word))
        lemmatized_array.append(new_text_array_1d)
    return lemmatized_array

def stem(text_array_2d):
    from nltk.stem.cistem import Cistem
    stemmer = Cistem()
    stemmed_array = []
    for text_array_1d in text_array_2d:
        new_text_array_1d = []
        for word in text_array_1d:
            new_text_array_1d.append(stemmer.stem(word)) # It also changes the characters to lowercase.
        stemmed_array.append(new_text_array_1d)
    return stemmed_array
