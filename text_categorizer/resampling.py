#!/usr/bin/python3
# coding=utf-8

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def RandomOverSample(X, y):
    ros = RandomOverSampler(sampling_strategy='auto', return_indices=False,
                random_state=None, ratio=None)
    X, y = ros.fit_resample(X, y)
    return X, y

def RandomUnderSample(X, y):
    rus = RandomUnderSampler(sampling_strategy='auto', return_indices=False,
                random_state=None, replacement=False, ratio=None)
    X, y = rus.fit_resample(X, y)
    return X, y
