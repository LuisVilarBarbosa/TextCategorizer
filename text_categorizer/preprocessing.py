#!/usr/bin/python3
# coding=utf-8

import stanfordnlp
import parameters

from tqdm import tqdm

def preprocess(docs):
    stanfordnlp_docs = generate_stanfordnlp_documents(docs)
    preprocessed_docs = stanfordnlp_process(stanfordnlp_docs)
    assert len(docs) == len(preprocessed_docs)
    for i in range(len(docs)):
        docs[i].update_stanfordnlp_document(preprocessed_docs[i])
    return docs

def generate_stanfordnlp_documents(docs):
    stanfordnlp_docs = []
    for doc in docs:
        text = doc.fields[parameters.EXCEL_COLUMN_WITH_TEXT_DATA]
        stanfordnlp_docs.append(stanfordnlp.Document(text))
    return stanfordnlp_docs

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

def stanfordnlp_process(stanfordnlp_docs):
    stanfordnlp_download()
    nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', lang=parameters.STANFORDNLP_LANGUAGE_PACKAGE, models_dir=parameters.STANFORDNLP_RESOURCES_DIR, use_gpu=parameters.STANFORDNLP_USE_GPU)
    processed_docs = []
    for doc in tqdm(iterable=stanfordnlp_docs, desc="Preprocessing", unit="doc"):
        processed_docs.append(nlp(doc))  # The lemma assigned by nlp() is in lowercase.
    return processed_docs
