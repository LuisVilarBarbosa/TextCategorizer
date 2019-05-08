#!/usr/bin/python3
# coding=utf-8

import signal
import stanfordnlp
import pickle_manager

from tqdm import tqdm
from logger import logger
from Parameters import Parameters

def preprocess(docs=None):
    return stanfordnlp_process(language_package=Parameters.STANFORDNLP_LANGUAGE_PACKAGE,
                               use_gpu=Parameters.STANFORDNLP_USE_GPU,
                               resource_dir=Parameters.STANFORDNLP_RESOURCES_DIR,
                               docs=docs,
                               text_data_field=Parameters.EXCEL_COLUMN_WITH_TEXT_DATA,
                               store_preprocessed_data=Parameters.TRAINING_MODE,
                               preprocessed_data_file=Parameters.PREPROCESSED_DATA_FILE)

def stanfordnlp_download(language_package, resource_dir):
    from os.path import isdir
    from os import listdir
    found = False
    if isdir(resource_dir):
        files = listdir(resource_dir)
        filename_start = ''.join([language_package, "_"])
        for file in files:
            if file.startswith(filename_start):
                found = True
                break
    if not found:
        stanfordnlp.download(language_package, resource_dir=resource_dir, confirm_if_exists=True, force=True)

_nlp = None
_stop = False

def stanfordnlp_process(language_package, use_gpu, resource_dir, text_data_field, store_preprocessed_data, docs=None, preprocessed_data_file=None):
    assert (docs is None) is store_preprocessed_data and store_preprocessed_data is (preprocessed_data_file is not None)
    global _nlp, _stop
    if _nlp is None:
        stanfordnlp_download(language_package=language_package, resource_dir=resource_dir)
        _nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', lang=language_package, models_dir=resource_dir, use_gpu=use_gpu)
    sig = signal.SIGINT
    old_handler = signal.signal(sig, signal_handler)
    logger.info("Press CTRL+C to stop the preprocessing phase. (The preprocessed documents will be stored.)")
    if docs is None:
        docs = pickle_manager.get_documents(preprocessed_data_file)
        total = pickle_manager.get_total_docs(preprocessed_data_file)
    else:
        total = len(docs)
    num_ignored = 0
    if store_preprocessed_data:
        pda = pickle_manager.PickleDumpAppend(total=total, filename=preprocessed_data_file)
    for doc in tqdm(iterable=docs, desc="Preprocessing", total=total, unit="doc", dynamic_ncols=True):
        if not _stop and doc.analyzed_sentences is None:
            text = doc.fields[text_data_field]
            try:
                stanfordnlp_doc = stanfordnlp.Document(text)
                stanfordnlp_doc_updated = _nlp(stanfordnlp_doc)
                doc.update(stanfordnlp_document=stanfordnlp_doc_updated, text_data_field=text_data_field)
            except Exception as e:
                print()
                logger.warning("Ignoring document number %s due to the following exception: %s" %  (doc.index, repr(e)))
                num_ignored = num_ignored + 1
        if store_preprocessed_data:
            pda.dump_append(doc)
    if store_preprocessed_data:
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
