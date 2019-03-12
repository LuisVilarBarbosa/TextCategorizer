#!/usr/bin/python3
# coding=utf-8

from setuptools import setup

setup(name='Text Categorizer',
      version='0.1',
      description='A tool to categorize textual data.',
      url='http://github.com/luisvilarbarbosa/DISS',
      author='Luís Barbosa',
      author_email='luisfernandobarbosa@live.com.pt',
      #license='MIT',
      packages=['text_categorizer'],
      install_requires=[
          'pandas==0.24.1',
          'xlrd==1.2.0',
          'nltk==3.4',
          'profilehooks==1.10.0',
          'numpy==1.16.1',
          'scikit-learn==0.20.3',
          'stanfordnlp==0.1.2',
          'tqdm==4.31.1',
          'conllu==1.2.3',
          'pynput==1.4',
      ],
      zip_safe=False)
