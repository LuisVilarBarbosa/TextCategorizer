#!/usr/bin/python3
# coding=utf-8

import stanfordnlp
import parameters

from tqdm import tqdm

def preprocess(str_list):
    docs = generate_documents(str_list)
    preprocessed_docs = stanfordnlp_process(docs)
    return preprocessed_docs

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

def stanfordnlp_process(docs):
    stanfordnlp_download()
    nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', lang=parameters.STANFORDNLP_LANGUAGE_PACKAGE, models_dir=parameters.STANFORDNLP_RESOURCES_DIR, use_gpu=parameters.STANFORDNLP_USE_GPU)
    processed_docs = []
    for doc in tqdm(iterable=docs, desc="Preprocessing", unit="doc"):
        processed_docs.append(nlp(doc))  # The lemma assigned by nlp() is in lowercase.
    return processed_docs
