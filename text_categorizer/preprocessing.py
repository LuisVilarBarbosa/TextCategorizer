#!/usr/bin/python3
# coding=utf-8

import numpy
import text_categorizer.parameters as parameters

def preprocess(text_array):
    tokenized_array = tokenize(text_array)
    filtered_array = filter(tokenized_array)
    lemmatized_array = lemmatize(filtered_array)
    stemmed_array = stem(lemmatized_array)
    return stemmed_array

def tokenize(text_array):
    from nltk.tokenize import MWETokenizer
    tokenizer = MWETokenizer([])
    new_text_array = numpy.array([])
    for text_elem in text_array:
        numpy.append(new_text_array, tokenizer.tokenize(text_elem.split()))
    return new_text_array

def filter(tokenized_array):
    from nltk import download
    download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop_words = set()
    for language in parameters.NLTK_STOP_WORDS_PACKAGES:
        stop_words = stop_words.union(set(stopwords.words(language)))
    # 'stop_words' contains text in lower case. That's not the case in 'tokenized_array'.
    filtered_array = numpy.array([])
    for array in tokenized_array:
        new_array = numpy.array([])
        for word in array:
            if not word in stop_words:
                numpy.append(new_array, word)
        numpy.append(filtered_array, new_array)
    return filtered_array

def lemmatize(filtered_array):
    from nltk import download
    download('wordnet', quiet=True)
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_array = numpy.array([])
    for array in filtered_array:
        new_array = numpy.array([])
        for word in array:
            numpy.append(new_array, lemmatizer.lemmatize(word))
        numpy.append(lemmatized_array, new_array)
    return lemmatized_array

def stem(lemmatized_array):
    from nltk.stem.cistem import Cistem
    stemmer = Cistem()
    stemmed_array = numpy.array([])
    for array in lemmatized_array:
        new_array = numpy.array([])
        for word in array:
            numpy.append(new_array, stemmer.stem(word)) # It also changes the characters to lowercase.
        numpy.append(stemmed_array, new_array)
    return stemmed_array
