import numpy as np
import pytest
from pandas import read_excel
from tests.utils import example_excel_file, generate_available_filename, remove_and_check
from text_categorizer import pickle_manager, train_test_split
from text_categorizer.FeatureExtractor import FeatureExtractor
from text_categorizer.functions import data_frame_to_document_list
from text_categorizer.Preprocessor import Preprocessor

def test_train_test_split():
    text_field = 'Example column'
    df = read_excel(example_excel_file)
    docs = data_frame_to_document_list(df)
    preprocessor = Preprocessor()
    preprocessor.preprocess(text_field, None, docs)
    ft = FeatureExtractor()
    corpus, classifications, _, _ = ft.prepare(text_field=text_field, class_field='Classification column', preprocessed_data_file=None, docs=docs, training_mode=False)
    test_size = 0.3
    preprocessed_data_file = generate_available_filename()
    force = False
    idxs_to_remove = [5]
    try:
        pickle_manager.dump_documents(docs, preprocessed_data_file)
        assert pickle_manager.get_docs_metadata(preprocessed_data_file) == {'total': 10}
        desired = {
            'total': 10,
            'test_size': test_size,
            'training_set_indexes': np.array([6, 1, 0, 2, 8, 3]),
            'test_set_indexes': np.array([7, 9, 4])
        }
        for my_force in [False, True]:
            train_test_split.train_test_split(corpus, classifications, test_size, preprocessed_data_file, my_force, idxs_to_remove)
            np.testing.assert_equal(pickle_manager.get_docs_metadata(preprocessed_data_file), desired)
        for key in ['test_size', 'training_set_indexes', 'test_set_indexes']:
            m = desired.copy()
            m[key] = None
            pickle_manager.set_docs_metadata(m, preprocessed_data_file)
            train_test_split.train_test_split(corpus, classifications, test_size, preprocessed_data_file, force, idxs_to_remove)
            np.testing.assert_equal(pickle_manager.get_docs_metadata(preprocessed_data_file), desired)
        for key, value in [('test_size', 0.2), ('training_set_indexes', np.array([1, 0, 2, 8, 3]))]:
            m = desired.copy()
            m[key] = value
            pickle_manager.set_docs_metadata(m, preprocessed_data_file)
            train_test_split.train_test_split(corpus, classifications, test_size, preprocessed_data_file, force, idxs_to_remove)
            np.testing.assert_equal(pickle_manager.get_docs_metadata(preprocessed_data_file), m)
    finally:
        remove_and_check(preprocessed_data_file)
    pass

def test_get_train_test():
    corpus = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    classifications = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    train_idxs = [9, 7, 6, 4, 3, 1, 0]
    test_idxs = [8, 5, 2]
    idxs_to_remove = [5, 3]
    invalid_train_idxs = train_idxs + train_idxs
    invalid_test_idxs = test_idxs + test_idxs
    with pytest.raises(AssertionError):
        train_test_split.get_train_test(corpus, classifications, invalid_train_idxs, test_idxs, idxs_to_remove)
    with pytest.raises(AssertionError):
        train_test_split.get_train_test(corpus, classifications, train_idxs, invalid_test_idxs, idxs_to_remove)
    corpus_train, corpus_test, classifications_train, classifications_test = train_test_split.get_train_test(corpus, classifications, train_idxs, test_idxs, idxs_to_remove)
    assert corpus_train == [19, 17, 16, 14, 11, 10]
    assert corpus_test == [18, 12]
    assert classifications_train == [29, 27, 26, 24, 21, 20]
    assert classifications_test == [28, 22]

def test__train_test_split():
    metadata1 = {}
    test_size = 0.3
    classifications = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
    idxs_to_remove = [5, 3]
    metadata2 = train_test_split._train_test_split(metadata1, test_size, classifications, idxs_to_remove)
    assert metadata1 is not metadata2
    assert metadata1 != metadata2
    assert metadata2['test_size'] == test_size
    assert np.array_equal(metadata2['training_set_indexes'], [4, 6, 9, 1, 8])
    assert np.array_equal(metadata2['test_set_indexes'], [7, 0, 2])

def test__is_stratified():
    classifications = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
    metadata = {'test_size': 0.3, 'training_set_indexes': [4, 6, 9, 1, 8], 'test_set_indexes': [7, 0, 2]}
    idxs_to_remove = [5, 3]
    assert train_test_split._is_stratified(classifications, metadata, idxs_to_remove)
    metadata = {'test_size': 0.3, 'training_set_indexes': [4, 6, 9, 0, 8], 'test_set_indexes': [7, 1, 2]}
    assert not train_test_split._is_stratified(classifications, metadata, idxs_to_remove)
