#!/usr/bin/python3
# coding=utf-8

import stanfordnlp
import pickle_manager

from tqdm import tqdm
from KeyboardListener import KeyboardListener
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
        stanfordnlp.download(Parameters.STANFORDNLP_LANGUAGE_PACKAGE, resource_dir=Parameters.STANFORDNLP_RESOURCES_DIR, confirm_if_exists=True, force=False)

_nlp = None
_stop = False

def stanfordnlp_process(docs=None):
    global _nlp, _stop
    if _nlp is None:
        stanfordnlp_download()
        _nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', lang=Parameters.STANFORDNLP_LANGUAGE_PACKAGE, models_dir=Parameters.STANFORDNLP_RESOURCES_DIR, use_gpu=Parameters.STANFORDNLP_USE_GPU)
    keyboard_listener = configure_keyboard_listener()
    if docs is None:
        docs = pickle_manager.get_documents()
        total = pickle_manager.get_total_docs()
    else:
        total = len(docs)
    num_ignored = 0
    if Parameters.TRAIN_MODE:
        pda = pickle_manager.PickleDumpAppend(total=total, filename=Parameters.PREPROCESSED_DATA_FILE)
    for doc in tqdm(iterable=docs, desc="Preprocessing", total=total, unit="doc"):
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
    if keyboard_listener is not None:
        keyboard_listener.stop()
    logger.warning("%s document(s) ignored." % num_ignored)
    if _stop:
        exit(0)

def on_press(key):
    from pynput.keyboard import Key
    if key == Key.esc:
        print()
        logger.info("Stopping the preprocessing phase.")
        global _stop
        _stop = True

def configure_keyboard_listener():
    keyboard_listener = None
    if Parameters.TRAIN_MODE:
        if KeyboardListener.available():
            keyboard_listener = KeyboardListener(on_press=on_press)
            logger.info("Press Esc to stop the preprocessing phase. (The preprocessed documents will    be stored.)")
        else:
            logger.info("Please, do not stop the program or some of the data might be lost.")
            logger.info("If you need to stop the preprocessing phase, press CTRL-C and restart the  program with the correct configuration.")
            logger.info(KeyboardListener.how_to_make_available())
            from time import sleep
            for _ in tqdm(iterable=range(30), desc="Waiting for cancellation order", unit="s"):
                sleep(1)
    return keyboard_listener
