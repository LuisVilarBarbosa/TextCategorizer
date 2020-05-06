import nltk
import re
import signal
from mosestokenizer import MosesSentenceSplitter, MosesTokenizer
from string import punctuation
from text_categorizer import pickle_manager
from text_categorizer.logger import logger
from text_categorizer.SpellChecker import SpellChecker
from text_categorizer.ui import get_documents, progress
from traceback import format_exc

class Preprocessor:
    stop_signals = [
        signal.SIGTERM,     # SIGTERM is sent by Docker on CTRL-C or on a call to 'docker stop'.
    ]

    def __init__(self, mosestokenizer_language_code="en", store_data=False, spell_checker_lang=None, n_jobs=1):
        self.mosestokenizer_language_code = mosestokenizer_language_code
        self.splitsents = MosesSentenceSplitter(self.mosestokenizer_language_code)
        self.tokenize = MosesTokenizer(self.mosestokenizer_language_code)
        nltk.download('wordnet', quiet=False)
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.stop = False
        self.store_data = store_data
        if spell_checker_lang is None:
            logger.info("The spell checker is disabled.")
            self.spell_checker = None
        else:
            logger.info("The spell checker is enabled for %s." % (spell_checker_lang))
            self.spell_checker = SpellChecker(language=spell_checker_lang, n_jobs=n_jobs)

    def preprocess(self, text_field, preprocessed_data_file=None, docs=None):
        if self.store_data:
            self._set_signal_handlers()
            logger.info("Send a SIGTERM signal to stop the preprocessing phase. (The preprocessed documents will be stored.)")
        description = "Preprocessing"
        if docs is None:
            docs = get_documents(preprocessed_data_file, description=description)
        else:
            docs = progress(iterable=docs, desc=description, unit="doc")
        if self.store_data:
            metadata = pickle_manager.get_docs_metadata(preprocessed_data_file)
            pda = pickle_manager.PickleDumpAppend(metadata=metadata, filename=preprocessed_data_file)
        token_to_lemma = dict()
        pattern = re.compile(r'\r\n|\r|\n')
        for doc in docs:
            if not self.stop and doc.analyzed_sentences.get(text_field) is None:
                text = doc.fields[text_field]
                text = pattern.sub("", text)
                sentences = self.splitsents([text])
                sentences = [self.tokenize(sent) for sent in sentences]
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
                doc.analyzed_sentences[text_field] = analyzed_sentences
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
