#!/usr/bin/python3
# coding=utf-8

import classifiers
import feature_extraction
import pickle_manager

from flask import Flask, jsonify, make_response, request, abort
from flask_httpauth import HTTPBasicAuth
from sys import argv
from Document import Document
from Parameters import Parameters
from Preprocessor import Preprocessor

app = Flask(__name__)
auth = HTTPBasicAuth()
BAD_REQUEST = 400
UNAUTHORIZED_ACCESS = 401
NOT_FOUND = 404
_preprocessor = None
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
    global _preprocessor
    if not request.json:
        abort(BAD_REQUEST)
    text = request.json.get('text')
    classifier = request.json.get('classifier')
    if type(text) is not str:
        abort(BAD_REQUEST, 'Invalid text')
    if type(classifier) is not str:
        abort(BAD_REQUEST, 'Invalid classifier')
    doc = Document(index=-1, fields=dict(), analyzed_sentences=None)
    doc.fields[Parameters.EXCEL_COLUMN_WITH_TEXT_DATA] = text
    doc.fields[Parameters.EXCEL_COLUMN_WITH_CLASSIFICATION_DATA] = None
    _preprocessor.preprocess([doc])
    X, _y, lemmas = feature_extraction.generate_X_y([doc])
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
    global _feature_weights
    features_dict = pickle_manager.load("features.pkl")
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

def main():
    global _preprocessor
    if len(argv) != 3:
        print("Usage: python3 text_categorizer/prediction_server.py <configuration file> <port>")
        quit()
    config_filename = argv[1]
    port = int(argv[2])
    limit_port = 1024
    if port <= limit_port:
        print("Please, indicate a port higher than %s." % (limit_port))
        quit()
    Parameters.load_configuration(config_filename, training_mode=False)
    _preprocessor = Preprocessor()
    app.run(host='0.0.0.0', port=port, debug=False) # host='0.0.0.0' allows access from any network.

if __name__ == '__main__':
    main()
