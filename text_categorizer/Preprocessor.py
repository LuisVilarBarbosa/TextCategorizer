#!/usr/bin/python3
# coding=utf-8

import nltk
import signal
from text_categorizer import pickle_manager
from text_categorizer.logger import logger
from text_categorizer.SpellChecker import SpellChecker
from text_categorizer.ui import get_documents, progress
from traceback import format_exc
from string import punctuation

# TODO: Allow to use both StanfordNLP and NLTK, choosing which one to use in the configuration file.
class Preprocessor:
    stop_signals = [
        signal.SIGINT,      # SIGINT is sent by CTRL-C.
        signal.SIGTERM,     # SIGTERM is sent by Docker on CTRL-C or on a call to 'docker stop'.
    ]

    def __init__(self, stanfordnlp_language_package="en", stanfordnlp_use_gpu=False, stanfordnlp_resources_dir="./stanfordnlp_resources", store_data=False, spell_checker_lang=None):
        quiet = True
        nltk.download('wordnet', quiet=quiet)
        nltk.download('punkt', quiet=quiet)
        nltk.download('averaged_perceptron_tagger', quiet=quiet)
        nltk.download('universal_tagset', quiet=quiet)
        self.language = stanfordnlp_language_package
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.stop = False
        self.store_data = store_data
        if spell_checker_lang is None:
            logger.info("The spell checker is disabled.")
            self.spell_checker = None
        else:
            logger.info("The spell checker is enabled for %s." % (spell_checker_lang))
            self.spell_checker = SpellChecker(language=spell_checker_lang)

    def preprocess(self, text_field, preprocessed_data_file=None, docs=None):
        return self._nltk_process(text_data_field=text_field, preprocessed_data_file=preprocessed_data_file, docs=docs)

    # TODO: PoS is not in use (only English and Russian are supported), so the removal of adjectives does not work correctly.
    def _nltk_process(self, text_data_field, preprocessed_data_file=None, docs=None):
        if self.store_data:
            self._set_signal_handlers()
            logger.info("Press CTRL+C to stop the preprocessing phase. (The preprocessed documents will be stored.)")
        description = "Preprocessing"
        if docs is None:
            docs = get_documents(preprocessed_data_file, description=description)
        else:
            docs = progress(iterable=docs, desc=description, unit="doc")
        if self.store_data:
            metadata = pickle_manager.get_docs_metadata(preprocessed_data_file)
            pda = pickle_manager.PickleDumpAppend(metadata=metadata, filename=preprocessed_data_file)
        token_to_lemma = dict()
        for doc in docs:
            if not self.stop and doc.analyzed_sentences is None:
                text = doc.fields[text_data_field]
                sentences = nltk.sent_tokenize(text)
                sentences = [nltk.word_tokenize(sent) for sent in sentences]
                if self.spell_checker is not None:
                    sentences = self.spell_checker.spell_check(sentences)
                analyzed_sentences = []
                for sent in sentences:
                    tokens = []
                    for word in sent:
                        token = word.lower()
                        lemma = token_to_lemma.get(token)
                        if lemma is None:
                            lemma = self.lemmatizer.lemmatize(token)
                            token_to_lemma[token] = lemma
                        token = {
                            'form': word,
                            'lemma': lemma,
                            'upostag': 'PUNCT' if lemma in punctuation else None
                        }
                        tokens.append(token)
                    analyzed_sentences.append(tokens)
                doc.analyzed_sentences = analyzed_sentences
            if self.store_data:
                pda.dump_append(doc)
        if self.store_data:
            pda.close()
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
        self.old_handlers.clear()
