#!/usr/bin/python3
# coding=utf-8

import stanfordnlp
import unittest
import classifiers
import functions
import pickle_manager
import preprocessing

class TestMethods(unittest.TestCase):
    texts = [
        "Them also them appear is saying is god bring, face given.",
        "In, winged tree gathering saw fifth grass, itself great and."
    ]
    docs = preprocessing.generate_documents(texts)
    expected_tokens = [
        ["Them", "also", "them", "appear", "is", "saying", "is", "god", "bring", ",", "face", "given", "."],
        ["In", ",", "winged", "tree", "gathering", "saw", "fifth", "grass", ",", "itself", "great", "and", "."]
    ]
    expected_lemmas = [
        ["also", "appear", "say", "god", "bring", ",", "face", "give", "."],
        [",", "wing", "tree", "gather", "see", "fifth", "grass", ",", "great", "."]
    ]

    def test_get_python_version(self):
        python_version = functions.get_python_version()
        self.assertEqual(type(python_version), int)
        self.assertGreaterEqual(python_version, 0)

    def test_preprocess(self):
        preprocessed_docs = preprocessing.preprocess(self.texts)
        tokens = []
        lemmas = []
        for doc in preprocessed_docs:
            doc_tokens = []
            doc_lemmas = []
            for sentence in doc.sentences:
                for token in sentence.tokens:
                    doc_tokens.append(token.text)
                for word in sentence.words:
                    doc_lemmas.append(word.lemma)
            tokens.append(doc_tokens)
            lemmas.append(doc_lemmas)
        self.assertEqual(tokens, self.expected_tokens)
        self.assertEqual(lemmas, self.expected_lemmas)

    def test_generate_documents(self):
        generated_documents = preprocessing.generate_documents(self.texts)
        self.assertGreater(len(generated_documents), 0)
        self.assertEqual(len(generated_documents), len(self.texts))
        for i in range(len(generated_documents)):
            doc = generated_documents[i]
            text = self.texts[i]
            self.assertIs(type(doc), stanfordnlp.Document)
            self.assertEqual(doc.text, text)

    def test_stanfordnlp_download(self):
        pass

    def test_stanfordnlp_process(self):
        processed_docs = preprocessing.stanfordnlp_process(self.docs)
        generated_tokens = []
        generated_lemmas = []
        for doc in processed_docs:
            doc_tokens = []
            doc_lemmas = []
            for sentence in doc.sentences:
                for token in sentence.tokens:
                    doc_tokens.append(token.text)
                for word in sentence.words:
                    doc_lemmas.append(word.lemma)
            generated_tokens.append(doc_tokens)
            generated_lemmas.append(doc_lemmas)
        self.assertEqual(generated_tokens, self.expected_tokens)
        expected_lemmas = [
            ["they", "also", "they", "appear", "be", "say", "be", "god", "bring", ",", "face", "give", "."],
            ["in", ",", "wing", "tree", "gather", "see", "fifth", "grass", ",", "itself", "great", "and", "."]
        ]
        self.assertEqual(generated_lemmas, expected_lemmas)        

    def test_filter(self):
        processed_docs = preprocessing.stanfordnlp_process(self.docs)
        filtered_docs = preprocessing.filter(processed_docs)
        filtered_lemmas = []
        for doc in filtered_docs:
            doc_lemmas = []
            for sentence in doc.sentences:
                for word in sentence.words:
                    doc_lemmas.append(word.lemma)
            filtered_lemmas.append(doc_lemmas)
        self.assertEqual(filtered_lemmas, self.expected_lemmas)

    def test_random_forest_classifier(self):
        # Improve test.
        from pandas.core.series import Series
        classification = Series(data=["I", "II"])
        classifiers.random_forest_classifier(self.docs, classification)

    def test_generate_corpus(self):
        pass

    def test_create_classification(self):
        pass

    def test_append_to_data_frame(self):
        from functions import append_to_data_frame
        from pandas import DataFrame
        data_frame = DataFrame()
        column_name = "test_append_to_data_frame"
        data_frame = append_to_data_frame(self.expected_tokens, data_frame, column_name)
        expected_column = [
            "Them,also,them,appear,is,saying,is,god,bring,,,face,given,.",
            "In,,,winged,tree,gathering,saw,fifth,grass,,,itself,great,and,."
        ]
        self.assertEqual(list(data_frame[column_name]), expected_column)

    def test_pickle_manager(self):
        from tempfile import mkstemp
        from os import close, unlink
        fd, file_path = mkstemp()
        pickle_manager.dump(obj=self.expected_tokens, filename=file_path)
        loaded_tokens = pickle_manager.load(filename=file_path)
        self.assertGreater(len(loaded_tokens), 0)
        self.assertEqual(self.expected_tokens, loaded_tokens)
        close(fd)
        unlink(file_path)

if __name__ == '__main__':
    from ui import verify_python_version
    verify_python_version()
    unittest.main()
