from os.path import exists
from tests.utils import generate_available_filename, remove_and_check
from text_categorizer.ContoPTParser import ContoPTParser

def test___init__():
    filename = generate_available_filename()
    try:
        assert not exists(filename)
        parser = ContoPTParser(filename)
        assert exists(filename)
        synonyms = ContoPTParser._load_synonyms(filename)
    finally:
        remove_and_check(filename)
    assert type(parser.synonyms) is dict
    assert len(parser.synonyms) > 0
    assert parser.synonyms == synonyms
    assert synonyms['adjudicatário'] == 'adjudicante'
    assert synonyms['melancolia'] == 'misantropia'
    assert synonyms['tristeza'] == 'misantropia'
