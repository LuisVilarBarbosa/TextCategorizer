import os
from hunspell import Hunspell
from requests import get

class SpellChecker:
    def __init__(self, language='en_US', hunspell_data_dir='./hunspell', n_jobs=1):
        SpellChecker.get_dict(language, hunspell_data_dir)
        self.hunspell = Hunspell(language, hunspell_data_dir=hunspell_data_dir, disk_cache_dir=os.path.join(hunspell_data_dir, 'cache'))
        self.hunspell.set_concurrency(n_jobs)
        self.substitutes = dict()

    def spell_check(self, tokenized_corpus_2d):
        tokens = [t for iterable in tokenized_corpus_2d for t in iterable]
        tokens = set(tokens) - self.substitutes.keys()
        suggestions = self.hunspell.bulk_suggest(tokens)
        self.substitutes.update(map(lambda kv: (kv[0], kv[0]) if not kv[1] else (kv[0], kv[1][0]), suggestions.items()))
        new_corpus = [[self.substitutes[token] for token in iterable] for iterable in tokenized_corpus_2d]
        return new_corpus
    
    @staticmethod
    def get_dict(language, data_dir):
        os.makedirs(data_dir, exist_ok=True)
        for ext in ['aff', 'dic']:
            path = os.path.join(data_dir, '%s.%s' % (language, ext))
            if not os.path.exists(path):
                r = get('https://raw.githubusercontent.com/LibreOffice/dictionaries/master/%s/%s.%s' % (language, language, ext))
                if r.status_code == 404:
                    l = language[0:language.find('_')]
                    r = get('https://raw.githubusercontent.com/LibreOffice/dictionaries/master/%s/%s.%s' % (l, language, ext))
                    r.raise_for_status()
                f = open(path, 'wb')
                f.write(r.content)
                f.close()
    
    def __del__(self):
        self.hunspell.save_cache() # For future program executions.
