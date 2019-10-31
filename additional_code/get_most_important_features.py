import json
import pickle
from pandas import DataFrame
from sys import argv

def load_pickle(path):
    input_file = open(path, 'rb')
    data = pickle.load(input_file)
    input_file.close()
    return data

def load_feature_weights(vectorizer, clf):
    feature_weights = set()
    if 'vocabulary_' in dir(vectorizer):
        features_dict = vectorizer.vocabulary_
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

def set_to_sorted_list(obj):
    if type(obj) is dict:
        fw = dict()
        for k in obj.keys():
            v = obj[k]
            assert type(v) is set
            v = list(v)
            v.sort(key=lambda item: -item[1])
            fw[k] = v
    elif type(obj) is set:
        fw = list(obj)
        fw.sort(key=lambda item: -item[1])
    else:
        raise ValueError(obj)
    return fw

def reduce_feature_weights(feature_weights, limit):
    if type(feature_weights) is dict:
        fw = dict()
        for k in feature_weights.keys():
            fw[k] = feature_weights[k][0:limit]
    elif type(feature_weights) is list:
        fw = feature_weights[0:limit]
    else:
        raise ValueError(feature_weights)
    return fw

def create_features_data_frame(feature_weights):
    if type(feature_weights) is dict:
        fw = feature_weights
    else:
        fw = {0: feature_weights}
    df = DataFrame()
    for k in fw.keys():
        df[k] = [item[0] for item in fw[k]]
    return df

def json_dump(obj, filename):
    f = open(filename, 'w')
    json.dump(obj, f)
    f.close()

def main():
    if len(argv) != 6:
        print("Usage: python3 %s <vectorizer_pickle_path> <classifier_pickle_path> <max_features> <JSON file name> <Excel file name>" % (argv[0]))
        quit()
    vectorizer_path = argv[1]
    classifier_path = argv[2]
    max_features = int(argv[3])
    json_file_name = argv[4]
    excel_file_name = argv[5]
    vectorizer = load_pickle(vectorizer_path)
    clf = load_pickle(classifier_path)
    fw = load_feature_weights(vectorizer, clf)
    fw = set_to_sorted_list(fw)
    fw = reduce_feature_weights(fw, limit=max_features)
    json_dump(fw, json_file_name)
    df = create_features_data_frame(fw)
    df.to_excel(excel_file_name)

if __name__ == '__main__':
    main()
