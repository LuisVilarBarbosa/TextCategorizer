import pytest
from sklearn.datasets import load_digits
from tests.test_classifiers import clfs
from text_categorizer.plot_roc import plot_roc

@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_plot_roc():
    for n_class in [2, 10]:
        X_test, y_test = load_digits(n_class=n_class, return_X_y=True)
        for f in clfs:
            clf = f(n_jobs=1, class_weight=None)
            clf.fit(X_test, y_test)
            plt = plot_roc(clf, X_test, y_test)
            plt.gcf().canvas.draw()
            plt.close()
