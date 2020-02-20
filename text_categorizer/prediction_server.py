from flask import Flask, jsonify, make_response, request, abort
from flask_httpauth import HTTPBasicAuth
from os.path import basename
from pandas import DataFrame
from text_categorizer import classifiers
from text_categorizer import pickle_manager
from text_categorizer.Document import Document
from text_categorizer.FeatureExtractor import FeatureExtractor
from text_categorizer.logger import logger
from text_categorizer.Parameters import Parameters
from text_categorizer.Preprocessor import Preprocessor

app = Flask(__name__)
auth = HTTPBasicAuth()
BAD_REQUEST = 400
UNAUTHORIZED_ACCESS = 401
NOT_FOUND = 404
_text_field = None
_class_field = None
_preprocessor = None
_feature_extractor = None
_feature_weights = dict()
_classifiers = dict()

@auth.get_password
def get_password(username):
    if username == 'admin':
        return 'admin'
    return None

@auth.error_handler
def unauthorized():
    return make_response(jsonify({'error': 'Unauthorized access'}), UNAUTHORIZED_ACCESS)

@app.route('/', methods=['POST'])
@auth.login_required
def predict():
    global _text_field, _class_field, _preprocessor, _feature_extractor, _classifiers
    if not request.json:
        abort(BAD_REQUEST)
    text = request.json.get('text')
    classifier = request.json.get('classifier')
    if type(text) is not str:
        abort(BAD_REQUEST, 'Invalid text')
    if type(classifier) is not str:
        abort(BAD_REQUEST, 'Invalid classifier')
    if basename(classifier) != classifier:
        abort(BAD_REQUEST, 'Invalid classifier')
    doc = Document(index=-1, fields=dict({_text_field: text, _class_field: None}), analyzed_sentences=dict())
    _preprocessor.preprocess(text_field=_text_field, docs=[doc])
    corpus, classifications, _idxs_to_remove, docs_lemmas = _feature_extractor.prepare(text_field=_text_field, class_field=_class_field, docs=[doc], training_mode=False)
    X, _y = _feature_extractor.generate_X_y(corpus, classifications, training_mode=False)
    try:
        clf = _classifiers.get(classifier)
        if clf is None:
            clf = pickle_manager.load("%s.pkl" % classifier)
            _classifiers[classifier] = clf
        y_predict_proba = clf.predict_proba(X)
        probabilities = classifiers.predict_proba_to_dicts(clf.classes_, y_predict_proba)[0]
        feature_weights = get_feature_weights(clf, docs_lemmas[0])
        probabilities = DataFrame({'probabilities': probabilities}).to_dict('dict')
        return jsonify({**probabilities, **feature_weights})
    except FileNotFoundError:
        abort(BAD_REQUEST, 'Invalid classifier model')

@app.errorhandler(NOT_FOUND)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), NOT_FOUND)

def get_feature_weights(clf, lemmas):
    global _feature_weights
    clf_name = clf.__class__.__name__
    all_feature_weights = _feature_weights.get(clf_name)
    lemmas_set = set(lemmas)
    if all_feature_weights is None:
        all_feature_weights = load_feature_weights(clf)
        _feature_weights[clf_name] = all_feature_weights
    if type(all_feature_weights) is set:
        fw = dict(filter(lambda item: item[0] in lemmas_set, all_feature_weights))
        feature_weights = DataFrame({'feature_weights': fw}).to_dict('dict')
    else:
        assert type(all_feature_weights) is dict
        feature_weights = dict()
        for c in all_feature_weights:
            fw = dict(filter(lambda item: item[0] in lemmas_set, all_feature_weights[c]))
            feature_weights[c] = fw
        feature_weights = DataFrame({'feature_weights': feature_weights}).to_dict('dict')
    return feature_weights

def load_feature_weights(clf):
    global _feature_extractor
    feature_weights = set()
    if 'vocabulary_' in dir(_feature_extractor.vectorizer):
        features_dict = _feature_extractor.vectorizer.vocabulary_
        features = sorted(features_dict, key=lambda k: features_dict[k])
        dir_clf = dir(clf)
        if "feature_importances_" in dir_clf:
            values = clf.feature_importances_
            assert len(features) == values.shape[0]
            feature_weights = set(zip(features, values))
        elif "coef_" in dir_clf:
            values = clf.coef_
            assert len(features) == values.shape[1]
            feature_weights = dict()
            clf_classes_ = clf.classes_
            if len(clf_classes_) == values.shape[0]:
                for i in range(len(clf_classes_)):
                    feature_weights[clf_classes_[i]] = set(zip(features, values[i]))
            else:
                for i in range(values.shape[0]):
                    feature_weights[i] = set(zip(features, values[i]))
    return feature_weights

def main(parameters, port):
    global _text_field, _class_field, _preprocessor, _feature_extractor
    limit_port = 1024
    if port <= limit_port:
        print("Please, indicate a port higher than %s." % (limit_port))
        quit()
    logger.disabled = True
    _text_field = parameters.excel_column_with_text_data
    _class_field = parameters.excel_column_with_classification_data
    _preprocessor = Preprocessor(stanfordnlp_language_package=parameters.stanfordnlp_language_package, stanfordnlp_use_gpu=parameters.stanfordnlp_use_gpu, stanfordnlp_resources_dir=parameters.stanfordnlp_resources_dir, store_data=False, spell_checker_lang=parameters.spell_checker_lang, n_jobs=parameters.number_of_jobs)
    _feature_extractor = FeatureExtractor(nltk_stop_words_package=parameters.nltk_stop_words_package, vectorizer_name=parameters.vectorizer, training_mode=False, feature_reduction=parameters.feature_reduction, document_adjustment_code=parameters.document_adjustment_code, remove_adjectives=parameters.remove_adjectives, synonyms_file=parameters.synonyms_file, n_jobs=parameters.number_of_jobs)
    app.run(host='0.0.0.0', port=port, debug=False) # host='0.0.0.0' allows access from any network.
