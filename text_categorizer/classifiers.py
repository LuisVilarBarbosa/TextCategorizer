import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def random_forest_classifier(docs, classification):
    corpus = generate_corpus(docs)
    X, y = create_classification(corpus, classification)
    # The code below is based on https://ehackz.com/2018/03/23/python-scikit-learn-random-forest-classifier-tutorial/ (accessed on 2019-02-26).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # The code below is based on https://www.kdnuggets.com/2017/12/general-approach-preprocessing-text-data.html (accessed on 2019-02-25).
    clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None,
                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None,
                random_state=None, verbose=0, warm_start=False, class_weight=None)
    clf.fit(X_train, y_train)
    #print(clf.feature_importances_)
    y_predict = clf.predict(X_test)
    # The code below is based on https://ehackz.com/2018/03/23/python-scikit-learn-random-forest-classifier-tutorial/ (accessed on 2019-02-25).
    return accuracy_score(y_test, y_predict, normalize=True)

def generate_corpus(docs):
    texts = []
    for doc in docs:
        text = ""
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.lemma != None:  # TODO: remove punctuation on preprocessing and remove this 'if' statement
                    text = ' '.join([text, word.lemma])
        texts.append(text)
    return texts

def create_classification(corpus, classification):
    # The code below is based on https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html (accessed on 2019-02-27).
    vectorizer = TfidfVectorizer(input='content', encoding='utf-8',
                decode_error='strict', strip_accents=None, lowercase=True,
                preprocessor=None, tokenizer=None, analyzer='word',
                stop_words=None, token_pattern=r'\S+',
                ngram_range=(1,1), max_df=1.0, min_df=1, max_features=None,
                vocabulary=None, binary=False, dtype=np.float64, norm='l2',
                use_idf=True, smooth_idf=True, sublinear_tf=False)
    X = vectorizer.fit_transform(corpus)
    y = classification
    #print(vectorizer.get_feature_names())
    #print(X.shape)
    return X, y
