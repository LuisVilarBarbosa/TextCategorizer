#!/usr/bin/python3
# coding=utf-8

import conllu
import nltk
import signal
import pickle_manager

from traceback import format_exc
from logger import logger
from string import punctuation
from ui import get_documents

# TODO: Allow to use both StanfordNLP and NLTK, choosing which one to use in the configuration file.
class Preprocessor:
    stop_signals = [
        signal.SIGINT,      # SIGINT is sent by CTRL-C.
        signal.SIGTERM,     # SIGTERM is sent by Docker on CTRL-C or on a call to 'docker stop'.
    ]

    def __init__(self, stanfordnlp_language_package="en", stanfordnlp_use_gpu=False, stanfordnlp_resources_dir="./stanfordnlp_resources", training_mode=True):
        quiet = True
        nltk.download('wordnet', quiet=quiet)
        nltk.download('punkt', quiet=quiet)
        nltk.download('averaged_perceptron_tagger', quiet=quiet)
        nltk.download('universal_tagset', quiet=quiet)
        self.language = stanfordnlp_language_package
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.stop = False
        self.training_mode = training_mode

    def preprocess(self, text_field, preprocessed_data_file=None, docs=None):
        return self._nltk_process(text_data_field=text_field, preprocessed_data_file=preprocessed_data_file, docs=docs)

    # TODO: PoS is invalid for non-English (only English and Russian are supported), so the removal of adjectives does not work correctly for non-English.
    def _nltk_process(self, text_data_field, preprocessed_data_file=None, docs=None):
        if self.training_mode:
            self._set_signal_handlers()
            logger.info("Press CTRL+C to stop the preprocessing phase. (The preprocessed documents will be stored.)")
        if docs is None:
            docs = get_documents(preprocessed_data_file, description="Preprocessing")
        if self.training_mode:
            metadata = pickle_manager.get_docs_metadata(preprocessed_data_file)
            pda = pickle_manager.PickleDumpAppend(metadata=metadata, filename=preprocessed_data_file)
        for doc in docs:
            if not self.stop and doc.analyzed_sentences is None:
                text = doc.fields[text_data_field]
                sentences = [nltk.pos_tag(nltk.word_tokenize(sent), tagset='universal', lang='eng') for sent in nltk.sent_tokenize(text)]
                conll = ''.join(nltk.parse.util.taggedsents_to_conll(sentences))
                doc.analyzed_sentences = conllu.parse(conll)
                for s in doc.analyzed_sentences:
                    for word in s:
                        lemma = self.lemmatizer.lemmatize(word['form'].lower())
                        word['lemma'] = lemma
                        if lemma in punctuation:
                            word['upostag'] = "PUNCT"
            if self.training_mode:
                pda.dump_append(doc)
        if self.training_mode:
            pda.close()
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
