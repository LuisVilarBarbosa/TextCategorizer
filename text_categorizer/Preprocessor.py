#!/usr/bin/python3
# coding=utf-8

import signal
import stanfordnlp
import pickle_manager

from tqdm import tqdm
from traceback import format_exc
from logger import logger
from Parameters import Parameters

class Preprocessor:
    stop_signals = [
        signal.SIGINT,      # SIGINT is sent by CTRL-C.
        signal.SIGTERM,     # SIGTERM is sent by Docker on CTRL-C or on a call to 'docker stop'.
    ]

    def __init__(self):
        language_package = Parameters.STANFORDNLP_LANGUAGE_PACKAGE
        use_gpu = Parameters.STANFORDNLP_USE_GPU
        resource_dir = Parameters.STANFORDNLP_RESOURCES_DIR
        Preprocessor._stanfordnlp_download(language_package=language_package, resource_dir=resource_dir)
        self.nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', lang=language_package, models_dir=resource_dir, use_gpu=use_gpu)
        self.stop = False

    def preprocess(self, docs=None):
        return self._stanfordnlp_process(docs=docs,
                                   text_data_field=Parameters.EXCEL_COLUMN_WITH_TEXT_DATA,
                                   training_mode=Parameters.TRAINING_MODE,
                                   store_preprocessed_data=Parameters.TRAINING_MODE,
                                   preprocessed_data_file=Parameters.PREPROCESSED_DATA_FILE)

    @staticmethod
    def _stanfordnlp_download(language_package, resource_dir):
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

    def _stanfordnlp_process(self, text_data_field, training_mode, store_preprocessed_data, docs=None, preprocessed_data_file=None):
        if training_mode:
            self._set_signal_handlers()
            logger.info("Press CTRL+C to stop the preprocessing phase. (The preprocessed documents will be stored.)")
        if docs is None:
            docs = pickle_manager.get_documents(preprocessed_data_file)
            total = pickle_manager.get_docs_metadata(Parameters.PREPROCESSED_DATA_FILE)['total']
        else:
            total = len(docs)
        num_ignored = 0
        if store_preprocessed_data:
            pda = pickle_manager.PickleDumpAppend(total=total, filename=preprocessed_data_file)
        for doc in tqdm(iterable=docs, desc="Preprocessing", total=total, unit="doc", dynamic_ncols=True):
            if not self.stop and doc.analyzed_sentences is None:
                text = doc.fields[text_data_field]
                try:
                    stanfordnlp_doc = stanfordnlp.Document(text)
                    stanfordnlp_doc_updated = self.nlp(stanfordnlp_doc)
                    doc.update(stanfordnlp_document=stanfordnlp_doc_updated, text_data_field=text_data_field)
                except Exception:
                    print()
                    logger.warning("Ignoring the document with index %s due to the following exception:\n%s" %  (doc.index, format_exc()))
                    num_ignored = num_ignored + 1
            if store_preprocessed_data:
                pda.dump_append(doc)
        if store_preprocessed_data:
            pda.close()
        logger.warning("%s document(s) ignored." % num_ignored)
        if training_mode:
            self._reset_signal_handlers()
        if self.stop:
            exit(0)

    def _signal_handler(self, sig, frame):
        if sig in Preprocessor.stop_signals:
            if not self.stop:
                print()
                logger.info("Stopping the preprocessing phase.")
                self.stop = True
    
    def _set_signal_handlers(self):
        self.old_handlers = dict()
        for sig in Preprocessor.stop_signals:
            self.old_handlers[sig] = signal.signal(sig, self._signal_handler)
    
    def _reset_signal_handlers(self):
        for sig, old_handler in self.old_handlers.items():
            signal.signal(sig, old_handler)
