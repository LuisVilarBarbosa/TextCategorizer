#!/usr/bin/python3
# coding=utf-8

import stanfordnlp
import parameters
import pickle_manager

from pynput.keyboard import Key, Listener
from tqdm import tqdm

def preprocess():
    return stanfordnlp_process()

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

def stanfordnlp_process():
    global _stop
    stanfordnlp_download()
    nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', lang=parameters.STANFORDNLP_LANGUAGE_PACKAGE, models_dir=parameters.STANFORDNLP_RESOURCES_DIR, use_gpu=parameters.STANFORDNLP_USE_GPU)
    with Listener(on_press=on_press) as listener:
        print("Press Esc to stop the preprocessing phase. (The preprocessed documents will be stored.)")
        tq = tqdm(desc="Preprocessing", total=pickle_manager.get_total_docs(), unit="doc")
        num_ignored = 0
        for path in pickle_manager.files_paths():
            docs = pickle_manager.load_documents(path)
            changed = False
            for doc in docs:
                if doc.analyzed_sentences is None:
                    text = doc.fields[parameters.EXCEL_COLUMN_WITH_TEXT_DATA]
                    try:
                        stanfordnlp_doc = stanfordnlp.Document(text)
                        stanfordnlp_doc_updated = nlp(stanfordnlp_doc)
                        doc.update_stanfordnlp_document(stanfordnlp_doc_updated)
                        changed = True
                    except Exception as e:
                        print()
                        print("Warning - Ignoring document number %s due to the following exception: %s" %  (doc.index, str(e)))
                        num_ignored = num_ignored + 1
                tq.update(1)
                if _stop:
                    if changed:
                        pickle_manager.dump_documents(docs, path)
                    exit(0)
            if changed:
                pickle_manager.dump_documents(docs, path)
        listener.stop()
        print("Warning - %s document(s) ignored." % num_ignored)

def on_press(key):
    if key == Key.esc:
        print()
        print("Stopping the preprocessing phase.")
        global _stop
        _stop = True
