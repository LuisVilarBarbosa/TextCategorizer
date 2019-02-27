#!/usr/bin/python3
# coding=utf-8

import unittest
import text_categorizer.classifiers as classifiers
import text_categorizer.functions as functions
import text_categorizer.preprocessing as preprocessing

class TestMethods(unittest.TestCase):
    text_array_1d = [
        "Them also them appear is saying is god bring, face given.",
        "In, winged tree gathering saw fifth grass, itself great and."
    ]
    text_array_2d = [
        ["Them", "also", "them", "appear", "is", "saying", "is", "god", "bring,", "face", "given."],
        ["In,", "winged", "tree", "gathering", "saw", "fifth", "grass,", "itself", "great", "and."]
    ]
   
    def test_get_python_version(self):
        python_version = functions.get_python_version()
        self.assertEqual(type(python_version), int)
        self.assertTrue(python_version >= 0)
    
    def test_preprocess(self):
        preprocessed_array = [
            ["them", "also", "appear", "saying", "god", "bring,", "fac", "given."],
            ["in,", "winged", "tree", "gathering", "saw", "fifth", "grass,", "grea", "and."]
        ]
        self.assertEqual(preprocessing.preprocess(self.text_array_1d), preprocessed_array)
    
    def test_tokenize(self):
        tokenized_array = [
            ["Them", "also", "them", "appear", "is", "saying", "is", "god", "bring,", "face", "given."],
            ["In,", "winged", "tree", "gathering", "saw", "fifth", "grass,", "itself", "great", "and."]
        ]
        self.assertEqual(preprocessing.tokenize(self.text_array_1d), tokenized_array)
    
    def test_filter(self):
        filtered_array = [
            ["Them", "also", "appear", "saying", "god", "bring,", "face", "given."],
            ["In,", "winged", "tree", "gathering", "saw", "fifth", "grass,", "great", "and."]
        ]
        self.assertEqual(preprocessing.filter(self.text_array_2d), filtered_array)
    
    def test_lemmatize(self):
        # Improve test.
        lemmatized_array = [
            ["Them", "also", "them", "appear", "is", "saying", "is", "god", "bring,", "face", "given."],
            ["In,", "winged", "tree", "gathering", "saw", "fifth", "grass,", "itself", "great", "and."]
        ]
        self.assertEqual(preprocessing.lemmatize(self.text_array_2d), lemmatized_array)
    
    def test_stem(self):
        stemmed_array = [
            ["them", "also", "them", "appear", "is", "saying", "is", "god", "bring,", "fac", "given."],
            ["in,", "winged", "tree", "gathering", "saw", "fifth", "grass,", "itself", "grea", "and."]
        ]
        self.assertEqual(preprocessing.stem(self.text_array_2d), stemmed_array)

    def test_random_forest_classifier(self):
        # Improve test.
        classifiers.random_forest_classifier()
    
    def test_append_to_data_frame(self):
        from functions import append_to_data_frame
        from pandas import DataFrame
        data_frame = DataFrame()
        column_name = "test_append_to_data_frame"
        data_frame = append_to_data_frame(self.text_array_2d, data_frame, column_name)
        expected_column = [
            "Them,also,them,appear,is,saying,is,god,bring,,face,given.",
            "In,,winged,tree,gathering,saw,fifth,grass,,itself,great,and."
        ]
        self.assertEqual(list(data_frame[column_name]), expected_column)

if __name__ == '__main__':
    from text_categorizer.ui import verify_python_version
    verify_python_version()
    unittest.main()
