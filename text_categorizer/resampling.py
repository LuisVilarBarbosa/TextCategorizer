#!/usr/bin/python3
# coding=utf-8

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from text_categorizer.constants import random_state

def RandomOverSample(X, y):
    ros = RandomOverSampler(sampling_strategy='auto', return_indices=False,
                random_state=random_state, ratio=None)
    X, y = ros.fit_resample(X, y)
    return X, y

def RandomUnderSample(X, y):
    rus = RandomUnderSampler(sampling_strategy='auto', return_indices=False,
                random_state=random_state, replacement=False, ratio=None)
    X, y = rus.fit_resample(X, y)
    return X, y
