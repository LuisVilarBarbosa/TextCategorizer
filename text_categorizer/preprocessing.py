#!/usr/bin/python3
# coding=utf-8

import signal
import stanfordnlp
import pickle_manager

from tqdm import tqdm
from logger import logger
from Parameters import Parameters

def preprocess(docs=None):
    return stanfordnlp_process(docs)

def stanfordnlp_download():
    from os.path import isdir
    from os import listdir
    found = False
    if isdir(Parameters.STANFORDNLP_RESOURCES_DIR):
        files = listdir(Parameters.STANFORDNLP_RESOURCES_DIR)
        filename_start = ''.join([Parameters.STANFORDNLP_LANGUAGE_PACKAGE, "_"])
        for file in files:
            if file.startswith(filename_start):
                found = True
                break
    if not found:
        stanfordnlp.download(Parameters.STANFORDNLP_LANGUAGE_PACKAGE, resource_dir=Parameters.STANFORDNLP_RESOURCES_DIR, confirm_if_exists=True, force=True)

_nlp = None
_stop = False

def stanfordnlp_process(docs=None):
    global _nlp, _stop
    if _nlp is None:
        stanfordnlp_download()
        _nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', lang=Parameters.STANFORDNLP_LANGUAGE_PACKAGE, models_dir=Parameters.STANFORDNLP_RESOURCES_DIR, use_gpu=Parameters.STANFORDNLP_USE_GPU)
    sig = signal.SIGINT
    old_handler = signal.signal(sig, signal_handler)
    logger.info("Press CTRL+C to stop the preprocessing phase. (The preprocessed documents will be stored.)")
    if docs is None:
        docs = pickle_manager.get_documents()
        total = pickle_manager.get_total_docs()
    else:
        total = len(docs)
    num_ignored = 0
    if Parameters.TRAIN_MODE:
        pda = pickle_manager.PickleDumpAppend(total=total, filename=Parameters.PREPROCESSED_DATA_FILE)
    for doc in tqdm(iterable=docs, desc="Preprocessing", total=total, unit="doc", dynamic_ncols=True):
        if not _stop and doc.analyzed_sentences is None:
            text = doc.fields[Parameters.EXCEL_COLUMN_WITH_TEXT_DATA]
            try:
                stanfordnlp_doc = stanfordnlp.Document(text)
                stanfordnlp_doc_updated = _nlp(stanfordnlp_doc)
                doc.update(stanfordnlp_doc_updated)
            except Exception as e:
                print()
                logger.warning("Ignoring document number %s due to the following exception: %s" %  (doc.index, repr(e)))
                num_ignored = num_ignored + 1
        if Parameters.TRAIN_MODE:
            pda.dump_append(doc)
    if Parameters.TRAIN_MODE:
        pda.close()
    logger.warning("%s document(s) ignored." % num_ignored)
    signal.signal(sig, old_handler)
    if _stop:
        exit(0)

def signal_handler(sig, frame):
    if sig == signal.SIGINT:
        global _stop
        if not _stop:
            print()
            logger.info("Stopping the preprocessing phase.")
            _stop = True
