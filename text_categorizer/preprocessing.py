#!/usr/bin/python3
# coding=utf-8

import stanfordnlp
import parameters

from time import time
from tqdm import tqdm

def preprocess(docs):
    return stanfordnlp_process(docs)

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
    processed_docs = docs.copy()
    t1 = time()
    for doc in tqdm(iterable=processed_docs, desc="Preprocessing", unit="doc"):
        if doc.analyzed_sentences is None:
            text = doc.fields[parameters.EXCEL_COLUMN_WITH_TEXT_DATA]
            stanfordnlp_doc = stanfordnlp.Document(text)
            stanfordnlp_doc_updated = nlp(stanfordnlp_doc)
            doc.update_stanfordnlp_document(stanfordnlp_doc_updated)
        t2 = time()
        if t2 - t1 >= parameters.PREPROCESSED_DATA_FILE_UPDATE_INTERVAL:
            dump_documents(processed_docs)
            t1 = t2
    dump_documents(processed_docs)
    return processed_docs

def dump_documents(docs):
    from jsonpickle_manager import dump
    dump(obj=docs, filename=parameters.PREPROCESSED_DATA_FILE)
