#!/usr/bin/python3
# coding=utf-8

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def RandomOverSample(X_train, y_train):
    ros = RandomOverSampler(sampling_strategy='auto', return_indices=False,
                random_state=None, ratio=None)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    return X_train, y_train

def RandomUnderSample(X_train, y_train):
    rus = RandomUnderSampler(sampling_strategy='auto', return_indices=False,
                random_state=None, replacement=False, ratio=None)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    return X_train, y_train
