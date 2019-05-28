#!/usr/bin/python3
# coding=utf-8

import classifiers
import pickle_manager

from flask import Flask, jsonify, make_response, request, abort
from flask_httpauth import HTTPBasicAuth
from Document import Document
from FeatureExtractor import FeatureExtractor
from logger import logger
from Parameters import Parameters
from Preprocessor import Preprocessor

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

@auth.get_password
def get_password(username):
    if username == 'admin':
        return 'admin'
    return None

@auth.error_handler
def unauthorized():
    return make_response(jsonify({'error': 'Unauthorized access'}), UNAUTHORIZED_ACCESS)

# TODO: Check if the files are in a valid directory.
@app.route('/', methods=['POST'])
@auth.login_required
def predict():
    global _text_field, _class_field, _preprocessor, _feature_extractor
    if not request.json:
        abort(BAD_REQUEST)
    text = request.json.get('text')
    classifier = request.json.get('classifier')
    if type(text) is not str:
        abort(BAD_REQUEST, 'Invalid text')
    if type(classifier) is not str:
        abort(BAD_REQUEST, 'Invalid classifier')
    doc = Document(index=-1, fields=dict({_text_field: text, _class_field: None}), analyzed_sentences=None)
    _preprocessor.preprocess(text_field=_text_field, docs=[doc])
    X, _y, lemmas = _feature_extractor.generate_X_y(class_field=_class_field, docs=[doc])
    try:
        clf = pickle_manager.load("%s.pkl" % classifier)
        y_predict_proba = clf.predict_proba(X)
        y_predict_classes = classifiers.predict_proba_to_predict_classes(clf.classes_, y_predict_proba)
        try:
            feature_weights = get_feature_weights(clf, lemmas)
            return jsonify({'prediction': y_predict_classes[0], 'feature_weights': feature_weights})
        except NotImplementedError:
            return jsonify({'prediction': y_predict_classes[0]})
    except FileNotFoundError:
        abort(BAD_REQUEST, 'Invalid classifier model')

@app.errorhandler(NOT_FOUND)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), NOT_FOUND)

def get_feature_weights(clf, lemmas):
    global _feature_weights
    clf_name = clf.__class__.__name__
    all_feature_weights = _feature_weights.get(clf_name)
    if all_feature_weights is None:
        load_feature_weights(clf)
        all_feature_weights = _feature_weights.get(clf_name)
    if type(all_feature_weights) is list:
        feature_weights = sorted(filter(lambda item: item[0] in lemmas, all_feature_weights), key=lambda item: item[1] * -1)
    else:
        assert type(all_feature_weights) is dict
        feature_weights = dict()
        for c in all_feature_weights:
            feature_weights[c] = sorted(filter(lambda item: item[0] in lemmas, all_feature_weights[c]), key=lambda item: item[1] * -1)
    return feature_weights

def load_feature_weights(clf):
    global _feature_extractor, _feature_weights
    features_dict = _feature_extractor.vectorizer.vocabulary_
    features = sorted(features_dict, key=lambda k: features_dict[k])
    if "feature_importances_" in dir(clf):
        values = clf.feature_importances_
        assert len(features) == values.shape[0]
        feature_weights = list(zip(features, values))
    elif "coef_" in dir(clf):
        values = clf.coef_
        assert len(features) == values.shape[1]
        feature_weights = dict()
        if len(clf.classes_) == values.shape[0]:
            for i in range(len(clf.classes_)):
                feature_weights[clf.classes_[i]] = list(zip(features, values[i]))
        else:
            for i in range(values.shape[0]):
                feature_weights[i] = list(zip(features, values[i].toarray()[0]))
    else:
        raise NotImplementedError
    clf_name = clf.__class__.__name__
    _feature_weights[clf_name] = feature_weights

def main(config_filename, port):
    global _text_field, _class_field, _preprocessor, _feature_extractor
    limit_port = 1024
    if port <= limit_port:
        print("Please, indicate a port higher than %s." % (limit_port))
        quit()
    logger.disabled = True
    parameters = Parameters(config_filename, training_mode=False)
    _text_field = parameters.excel_column_with_text_data
    _class_field = parameters.excel_column_with_classification_data
    _preprocessor = Preprocessor(stanfordnlp_language_package=parameters.stanfordnlp_language_package, stanfordnlp_use_gpu=parameters.stanfordnlp_use_gpu, stanfordnlp_resources_dir=parameters.stanfordnlp_resources_dir, training_mode=parameters.training_mode)
    _feature_extractor = FeatureExtractor(nltk_stop_words_package=parameters.nltk_stop_words_package, vectorizer_name=parameters.vectorizer, training_mode=parameters.training_mode, use_lda=parameters.use_lda, document_adjustment_code=parameters.document_adjustment_code, remove_adjectives=parameters.remove_adjectives, synonyms_file=parameters.synonyms_file, features_file=parameters.features_file)
    app.run(host='0.0.0.0', port=port, debug=False) # host='0.0.0.0' allows access from any network.
