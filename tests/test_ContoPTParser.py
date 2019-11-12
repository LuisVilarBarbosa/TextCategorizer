import io
import requests
from tests.utils import create_temporary_file, remove_and_check
from text_categorizer.ContoPTParser import ContoPTParser
from zipfile import ZipFile

def test___init__():
    r = requests.get(url='http://ontopt.dei.uc.pt/recursos/CONTO.PT.01.zip', stream=True)
    with ZipFile(io.BytesIO(r.content)) as archive:
        data = archive.read('contopt_0.1_r2_c0.0.txt')
        try:
            path = create_temporary_file(content=data, text=False)
            parser = ContoPTParser(filename=path)
            synonyms = ContoPTParser._load_synonyms(path)
        finally:
            remove_and_check(path)
        assert type(parser.synonyms) is dict
        assert len(parser.synonyms) > 0
        assert parser.synonyms == synonyms

def test__load_synonyms():
    pass
