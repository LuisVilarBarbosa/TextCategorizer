#!/usr/bin/python3
# coding=utf-8

import stanfordnlp
import parameters

def preprocess(str_list):
    docs = generate_documents(str_list)
    tokenized_docs = tokenize(docs)
    lemmatized_docs = lemmatize(tokenized_docs)
    filtered_docs = filter(lemmatized_docs)
    return filtered_docs

def generate_documents(str_list):
    docs = []
    for text in str_list:
        docs.append(stanfordnlp.Document(text))
    return docs

def stanfordnlp_download():
    from os.path import isdir
    from os import listdir
    found = False
    if isdir(parameters.STANFORDNLP_RESOURCES_DIR):
        files = listdir(parameters.STANFORDNLP_RESOURCES_DIR)
        filename_start = ''.join([parameters.STANFORDNLP_LANGUAGE_PACKAGE, "_"])
        for file in files:
            if file.startswith(filename_start):
                found = True
                break
    if not found:
        stanfordnlp.download(parameters.STANFORDNLP_LANGUAGE_PACKAGE, resource_dir=parameters.STANFORDNLP_RESOURCES_DIR, confirm_if_exists=True, force=False)

def tokenize(docs):
    stanfordnlp_download()
    nlp = stanfordnlp.Pipeline(processors='tokenize,mwt', lang=parameters.STANFORDNLP_LANGUAGE_PACKAGE, models_dir=parameters.STANFORDNLP_RESOURCES_DIR, use_gpu=parameters.STANFORDNLP_USE_GPU)
    tokenized_docs = []
    for doc in docs:
        tokenized_docs.append(nlp(doc))
    return tokenized_docs

def lemmatize(docs):
    stanfordnlp_download()
    nlp = stanfordnlp.Pipeline(processors='pos,lemma', lang=parameters.STANFORDNLP_LANGUAGE_PACKAGE, models_dir=parameters.STANFORDNLP_RESOURCES_DIR, use_gpu=parameters.STANFORDNLP_USE_GPU)
    lemmatized_docs = []
    for doc in docs:
        lemmatized_docs.append(nlp(doc))  # The lemma assigned by nlp() is in lowercase.
    return lemmatized_docs

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
            for word in sentence.words:
                if word.lemma in stop_words:  # 'word.lemma' is in lowercase.
                    sentence.words.remove(word)
    return filtered_docs
