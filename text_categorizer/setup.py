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
          'pandas',
          'xlrd',
          'nltk',
          'profilehooks',
          'numpy',
          'sklearn',
          'stanfordnlp',
          'conllu',
          'jsonpickle',
      ],
      zip_safe=False)
