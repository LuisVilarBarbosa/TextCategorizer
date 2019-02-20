#!/usr/bin/python3
# coding=utf-8

import text_categorizer.parameters as parameters

def preprocess(text_array):
    tokenized_array = tokenize(text_array)
    filtered_array = filter(tokenized_array)
    return filtered_array

def tokenize(text_array):
    from nltk.tokenize import MWETokenizer
    tokenizer = MWETokenizer([])
    new_text_array = []
    for text_elem in text_array:
        new_text_array.append(tokenizer.tokenize(text_elem.split()))
    return new_text_array

def filter(tokenized_array):
    from nltk import download
    download('stopwords')
    from nltk.corpus import stopwords
    stop_words = set()
    for language in parameters.NLTK_STOP_WORDS_PACKAGES:
        stop_words.union(set(stopwords.words(language)))
    filtered_array = []
    for array in tokenized_array:
        new_array = []
        for word in array:
            if not word in stop_words:
                new_array.append(word)
        filtered_array.append(new_array)
    return filtered_array
