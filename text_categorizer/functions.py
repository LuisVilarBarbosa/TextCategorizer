#!/usr/bin/python3
# coding=utf-8

import pandas as pd
import time
from importlib.util import spec_from_file_location, module_from_spec
from os import path
from pandas import DataFrame
from sklearn.metrics import classification_report
from sys import version
from text_categorizer.Document import Document

def get_python_version():
    version_array = [int(n) for n in version[:version.find(" ")].split(".")]
    return version_array

def append_to_data_frame(array_2d, data_frame, column_name):
    new_data_frame = data_frame.copy()
    idx = len(new_data_frame.columns)
    new_column = []
    for array_1d in array_2d:
        new_column.append(','.join(array_1d))
    new_data_frame.insert(loc=idx, column=column_name, value=new_column, allow_duplicates=False)
    return new_data_frame

def data_frame_to_document_list(data_frame):
    documents = []
    for i in range(len(data_frame)):
        d = Document.from_data_frame(data_frame=data_frame, index=i)
        documents.append(d)
    return documents

def load_module(filename):
    name = path.splitext(path.basename(filename))[0]
    spec = spec_from_file_location(name, filename)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def predictions_to_data_frame(predictions_dict):
    predictions = predictions_dict.copy()
    y_true = predictions.pop('y_true')
    data = dict()
    for k, y_pred in predictions.items():
        clf = k[len('y_pred_'):]
        report = classification_report(y_true, y_pred, output_dict=True)
        for label in report.keys():
            for metric in report[label].keys():
                col = '%s %s %s' % (metric, clf, label)
                data[col] = report[label][metric]
    df = DataFrame([data])
    return df

def parameters_to_data_frame(parameters_dict):
    p = parameters_dict.copy()
    for k in p.keys():
        if p[k] is None:
            p[k] = 'None'
    col_param = [
        ['Excel file', 'excel_file'],
        ['Text column', 'excel_column_with_text_data'],
        ['Label column', 'excel_column_with_classification_data'],
        ['n_jobs', 'number_of_jobs'],
        ['Preprocessed data file', 'preprocessed_data_file'],
        ['Preprocess data', 'preprocess_data'],
        ['StanfordNLP language package', 'stanfordnlp_language_package'],
        ['StanfordNLP use gpu', 'stanfordnlp_use_gpu'],
        ['StanfordNLP resources dir', 'stanfordnlp_resources_dir'],
        ['Spell checker language', 'spell_checker_lang'],
        ['NLTK stop words package', 'nltk_stop_words_package'],
        ['Document adjustment code', 'document_adjustment_code'],
        ['Vectorizer', 'vectorizer'],
        ['Feature reduction', 'feature_reduction'],
        ['Remove adjectives', 'remove_adjectives'],
        ['Synonyms file', 'synonyms_file'],
        ['Vectorizer file', 'vectorizer_file'],
        ['Accepted probabilities', 'set_num_accepted_probs'],
        ['Test size', 'test_subset_size'],
        ['Force subsets regeneration', 'force_subsets_regeneration'],
        ['Resampling', 'resampling'],
        ['Class weights', 'class_weights'],
        ['Generate ROC plots', 'generate_roc_plots']
    ]
    assert len(p) - 1 == len(col_param)
    columns = list(map(lambda item: item[0], col_param))
    params = list(map(lambda item: p[item[1]], col_param))
    df = DataFrame(data=[params], columns=columns)
    return df

def generate_report(execution_info, parameters_dict, predictions_dict, excel_file='report.xlsx'):
    try:
        df1 = pd.read_excel(excel_file)
    except FileNotFoundError:
        df1 = pd.DataFrame()
    p = parameters_dict.copy()
    for accepted_probs in [1]: #parameters_dict['set_num_accepted_probs']:
        p['set_num_accepted_probs'] = accepted_probs
        parameters_df = parameters_to_data_frame(p)
        predictions_df = predictions_to_data_frame(predictions_dict)
        df2 = pd.concat(objs=[execution_info, parameters_df, predictions_df], axis=1, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=True)
        df1 = pd.concat(objs=[df1, df2], axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=True)
    df1.to_excel(excel_file, index=False)
    return df1

def get_local_time_str(time_tuple=None):
    if time_tuple is None:
        time_tuple = time.localtime()
    return time.strftime('%Y-%m-%d %H:%M:%S %z %Z', time_tuple)
