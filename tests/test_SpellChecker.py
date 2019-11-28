import os
import pytest
import requests
from itertools import zip_longest
from shutil import rmtree
from tempfile import mkdtemp
from tests.utils import generate_available_filename
from text_categorizer.SpellChecker import SpellChecker
from time import time

def test___init__():
    data_dir = mkdtemp()
    cache_dir = os.path.join(data_dir, 'cache')
    assert not os.path.exists(cache_dir)
    assert not os.listdir(data_dir)
    for i, lang, max_threads in [(0, 'en_US', 1), (1, 'pt_PT', 4)]:
        aff = os.path.join(data_dir, '%s.aff' % (lang))
        dic = os.path.join(data_dir, '%s.dic' % (lang))
        assert not os.path.exists(aff)
        assert not os.path.exists(dic)
        if i == 0:
            sc  = SpellChecker(hunspell_data_dir=data_dir)
        elif i == 1:
            sc = SpellChecker('pt_PT', data_dir, 4)
        else:
            pytest.fail()
        assert sc.hunspell.lang == lang
        assert sc.hunspell._hunspell_dir == data_dir
        assert sc.hunspell.max_threads == max_threads
        assert sc.substitutes == dict()
        assert os.path.exists(aff)
        assert os.path.exists(dic)
    assert os.path.exists(cache_dir)
    with pytest.raises(requests.HTTPError):
        SpellChecker('pt_NotExists', data_dir)
    rmtree(data_dir)

def test_spell_check():
    data_dir = mkdtemp()
    corpus = [
        ['accomodation ', 'accommodation', 'adress', 'address'],
        ['wether', 'wEatHer', 'lenght'],
        ['length', 'recieve', 'reCeiVe'],
        ['collegue', 'colLeague', 'COffee', 'CAfee']
    ]
    expected_corpus = [
        ['accommodation', 'accommodation', 'dress', 'address'],
        ['ether', 'weather', 'length'],
        ['length', 'receive', 'receive'],
        ['college', 'col League', 'Coffee', 'CA fee']
    ]
    flatten = lambda l: [item for sublist in l for item in sublist]
    substitutes = dict(zip_longest(flatten(corpus), flatten(expected_corpus)))
    sc = SpellChecker(hunspell_data_dir=data_dir)
    assert sc.substitutes == dict()
    t1 = time()
    new_corpus = sc.spell_check(corpus)
    t1 = time() - t1
    assert corpus is not new_corpus
    assert corpus != new_corpus
    assert new_corpus == expected_corpus
    assert sc.substitutes == substitutes
    t2 = time()
    new_corpus = sc.spell_check(corpus)
    t2 = time() - t2
    assert new_corpus == expected_corpus
    assert t2 < t1 / 10
    assert sc.substitutes == substitutes
    new_corpus = sc.spell_check([['abcense']])
    assert new_corpus == [['absence']]
    substitutes['abcense'] = 'absence'
    assert sc.substitutes == substitutes
    rmtree(data_dir)

def test_get_dict():
    for data_dir in [mkdtemp(), generate_available_filename()]:
        for lang in ['en_US', 'pt_PT']:
            aff_file = os.path.join(data_dir, '%s.aff' % (lang))
            dic_file = os.path.join(data_dir, '%s.dic' % (lang))
            assert not os.path.exists(aff_file)
            assert not os.path.exists(dic_file)
            SpellChecker.get_dict(lang, data_dir)
            assert os.path.exists(aff_file)
            assert os.path.exists(dic_file)
            aff_mtime = os.path.getmtime(aff_file)
            dic_mtime = os.path.getmtime(dic_file)
            SpellChecker.get_dict(lang, data_dir)
            assert aff_mtime == os.path.getmtime(aff_file)
            assert dic_mtime == os.path.getmtime(dic_file)
        with pytest.raises(requests.HTTPError):
            SpellChecker.get_dict('pt_NotExists', data_dir) 
        rmtree(data_dir)

def test___del__():
    data_dir = mkdtemp()
    cache_dir = os.path.join(data_dir, 'cache')
    sc = SpellChecker(hunspell_data_dir=data_dir)
    assert not os.path.exists(cache_dir)
    del(sc)
    assert pytest.approx(os.path.getmtime(cache_dir)) == pytest.approx(time())
    assert len(os.listdir(cache_dir)) == 2
    rmtree(data_dir)
