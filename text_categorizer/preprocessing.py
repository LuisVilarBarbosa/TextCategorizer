#!/usr/bin/python3
# coding=utf-8

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
        stop_words = stop_words.union(set(stopwords.words(language)))
    # 'stop_words' contains text in lower case. That's not the case in 'tokenized_array'.
    filtered_array = []
    for array in tokenized_array:
        new_array = []
        for word in array:
            if not word in stop_words:
                new_array.append(word)
        filtered_array.append(new_array)
    return filtered_array

def lemmatize(filtered_array):
    from nltk import download
    download('wordnet')
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_array = []
    for array in filtered_array:
        new_array = []
        for word in array:
            new_array.append(lemmatizer.lemmatize(word))
        lemmatized_array.append(new_array)
    return lemmatized_array

def stem(lemmatized_array):
    from nltk.stem.cistem import Cistem
    stemmer = Cistem()
    stemmed_array = []
    for array in lemmatized_array:
        new_array = []
        for word in array:
            new_array.append(stemmer.stem(word)) # It also changes the characters to lowercase.
        stemmed_array.append(new_array)
    return stemmed_array
