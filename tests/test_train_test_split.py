import pytest
from text_categorizer import train_test_split

def test_train_test_split():
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
    pass

def test__is_stratified():
    pass
