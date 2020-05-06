import pandas

class Document:
    def __init__(self, index, fields, analyzed_sentences):
        self.index = index
        self.fields = fields
        self.analyzed_sentences = analyzed_sentences

    @staticmethod
    def from_data_frame(data_frame, index):
        assert type(data_frame) is pandas.DataFrame
        assert type(index) is int
        fields = dict()
        columns = data_frame.columns
        for i in range(len(columns)):
            fields[columns[i]] = data_frame.iloc[index, i]
        return Document(index=index, fields=fields, analyzed_sentences=dict())

    def copy(self):
        if self.analyzed_sentences is None:
            analyzed_sentences = None
        else:
            analyzed_sentences = self.analyzed_sentences.copy()
        return Document(self.index, self.fields.copy(), analyzed_sentences)

    def __repr__(self):
        return "%s: %s" % (self.__class__.__name__, self.__dict__)
