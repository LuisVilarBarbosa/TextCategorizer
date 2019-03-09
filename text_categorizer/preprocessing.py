#!/usr/bin/python3
# coding=utf-8

import stanfordnlp
import parameters

from pynput.keyboard import Key, Listener
from tqdm import tqdm
from jsonpickle_manager import dump_document

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

_stop = False

def stanfordnlp_process(docs):
    global _stop
    stanfordnlp_download()
    nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', lang=parameters.STANFORDNLP_LANGUAGE_PACKAGE, models_dir=parameters.STANFORDNLP_RESOURCES_DIR, use_gpu=parameters.STANFORDNLP_USE_GPU)
    processed_docs = docs.copy()
    with Listener(on_press=on_press) as listener:
        print("Each preprocessed document is being stored as soon as it is ready.")
        print("Press Esc to stop the preprocessing phase.")
        i = 0
        for doc in tqdm(iterable=processed_docs, desc="Preprocessing", unit="doc"):
            i = i + 1
            if doc.analyzed_sentences is None:
                text = doc.fields[parameters.EXCEL_COLUMN_WITH_TEXT_DATA]
                try:
                    stanfordnlp_doc = stanfordnlp.Document(text)
                    stanfordnlp_doc_updated = nlp(stanfordnlp_doc)
                    doc.update_stanfordnlp_document(stanfordnlp_doc_updated)
                    dump_document(doc, i)
                except AssertionError:
                    print()
                    print("Warning: Ignoring document number %s due to error on StanfordNLP." % i)
            if _stop:
                exit(0)
        listener.stop()
        return processed_docs

def on_press(key):
    if key == Key.esc:
        print()
        print("Stopping the preprocessing phase.")
        global _stop
        _stop = True
