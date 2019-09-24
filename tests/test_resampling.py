from sklearn.datasets import load_digits
from text_categorizer.resampling import RandomOverSample, RandomUnderSample

def test_RandomOverSample():
    X1, y1 = load_digits(n_class=10, return_X_y=True)
    X2, y2 = RandomOverSample(X1, y1)
    assert X1.shape[0] < X2.shape[0]
    assert X1.shape[1] == X2.shape[1]
    for elem in X2:
        assert elem in X1
    for elem in y2:
        assert elem in y1

def test_RandomUnderSample():
    X1, y1 = load_digits(n_class=10, return_X_y=True)
    X2, y2 = RandomUnderSample(X1, y1)
    assert X1.shape[0] >= X2.shape[0]
    assert X1.shape[1] == X2.shape[1]
    for elem in X2:
        assert elem in X1
    for elem in y2:
        assert elem in y1
