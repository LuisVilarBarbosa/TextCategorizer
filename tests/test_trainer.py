import pytest
from text_categorizer import trainer

def test_get_train_test():
    corpus = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    classifications = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    train_idxs = [0, 1, 3, 4, 6, 7, 9]
    test_idxs = [2, 5, 8]
    idxs_to_remove = [3, 5]
    invalid_train_idxs = train_idxs + train_idxs
    invalid_test_idxs = test_idxs + test_idxs
    with pytest.raises(AssertionError):
        trainer.get_train_test(corpus, classifications, invalid_train_idxs, test_idxs, idxs_to_remove)
    with pytest.raises(AssertionError):
        trainer.get_train_test(corpus, classifications, train_idxs, invalid_test_idxs, idxs_to_remove)
    corpus_train, corpus_test, classifications_train, classifications_test = trainer.get_train_test(corpus, classifications, train_idxs, test_idxs, idxs_to_remove)
    assert corpus_train == [10, 11, 14, 16, 17, 19]
    assert corpus_test == [12, 18]
    assert classifications_train == [20, 21, 24, 26, 27, 29]
    assert classifications_test == [22, 28]

def test_load_20newsgroups():
    pass

def test_main():
    pass
