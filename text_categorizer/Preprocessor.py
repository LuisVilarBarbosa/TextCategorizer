#!/usr/bin/python3
# coding=utf-8

import signal
import stanfordnlp
import pickle_manager

from traceback import format_exc
from logger import logger
from ui import get_documents

class Preprocessor:
    stop_signals = [
        signal.SIGINT,      # SIGINT is sent by CTRL-C.
        signal.SIGTERM,     # SIGTERM is sent by Docker on CTRL-C or on a call to 'docker stop'.
    ]

    def __init__(self, stanfordnlp_language_package="en", stanfordnlp_use_gpu=False, stanfordnlp_resources_dir="./stanfordnlp_resources", training_mode=True):
        Preprocessor._stanfordnlp_download(language_package=stanfordnlp_language_package, resource_dir=stanfordnlp_resources_dir)
        self.nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma', lang=stanfordnlp_language_package, models_dir=stanfordnlp_resources_dir, use_gpu=stanfordnlp_use_gpu)
        self.stop = False
        self.training_mode = training_mode

    def preprocess(self, text_field, preprocessed_data_file=None, docs=None):
        return self._stanfordnlp_process(text_data_field=text_field, preprocessed_data_file=preprocessed_data_file, docs=docs)

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

    def _stanfordnlp_process(self, text_data_field, preprocessed_data_file=None, docs=None):
        if self.training_mode:
            self._set_signal_handlers()
            logger.info("Press CTRL+C to stop the preprocessing phase. (The preprocessed documents will be stored.)")
        if docs is None:
            docs = get_documents(preprocessed_data_file, description="Preprocessing")
        num_ignored = 0
        if self.training_mode:
            metadata = pickle_manager.get_docs_metadata(preprocessed_data_file)
            pda = pickle_manager.PickleDumpAppend(metadata=metadata, filename=preprocessed_data_file)
        for doc in docs:
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
            if self.training_mode:
                pda.dump_append(doc)
        if self.training_mode:
            pda.close()
        if num_ignored > 0:
            logger.warning("%s document(s) ignored." % num_ignored)
        if self.training_mode:
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
