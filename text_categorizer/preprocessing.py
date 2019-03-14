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
        for filename in pickle_manager.filenames():
            docs = pickle_manager.load_documents(filename)            
            pda = pickle_manager.PickleDumpAppend(filename)
            for d in docs:
                if not _stop and d.analyzed_sentences is None:
                    doc = d.copy()
                    text = doc.fields[parameters.EXCEL_COLUMN_WITH_TEXT_DATA]
                    try:
                        stanfordnlp_doc = stanfordnlp.Document(text)
                        stanfordnlp_doc_updated = nlp(stanfordnlp_doc)
                        doc.update_stanfordnlp_document(stanfordnlp_doc_updated)
                    except Exception as e:
                        print()
                        print("Warning - Ignoring document number %s due to the following exception: %s" %  (doc.index, str(e)))
                        num_ignored = num_ignored + 1
                    pda.dump_append(doc)
                else:
                    pda.dump_append(d)
                tq.update(1)
            pda.close()
        listener.stop()
        print("Warning - %s document(s) ignored." % num_ignored)
        if _stop:
            exit(0)

def on_press(key):
    if key == Key.esc:
        print()
        print("Stopping the preprocessing phase.")
        global _stop
        _stop = True
