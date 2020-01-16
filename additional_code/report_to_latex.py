import pandas as pd
from collections import OrderedDict
from sys import argv

mapping = OrderedDict()
mapping['DummyClassifier'] = 'Random (stratified)'
mapping['BernoulliNB'] = 'Bernoulli NB'
mapping['MultinomialNB'] = 'Multinomial NB'
mapping['ComplementNB'] = 'Complement NB'
mapping['KNeighborsClassifier'] = 'K-Neighbors'
mapping['LinearSVC'] = 'SVM (linear)'
mapping['SGDClassifier'] = 'SGD'
mapping['DecisionTreeClassifier'] = 'Decision Tree'
mapping['ExtraTreeClassifier'] = 'Extra Tree'
mapping['RandomForestClassifier'] = 'Random Forests'
mapping['BaggingClassifier'] = 'Bagging'

def process_dataframe(df, keys):
    cols = [c for c in df.columns if all([k in c for k in keys])]
    df = df[cols]
    return df

def dataframe_to_latex(df_micro, df_macro):
    assert len(df_micro.columns) == 1 and len(df_macro.columns) == 1
    df = pd.DataFrame(index=mapping.values())
    df = pd.concat([df, df_micro, df_macro], axis=1, sort=False)
    latex = df.to_latex(header=['Acc', 'Macro-F1'], float_format="{:0.4f}".format)
    return latex

def rename_indexes(df):
    idxs = dict()
    for idx in df.index:
        parts = idx.split(' ')
        assert len(parts) >= 3
        assert parts[0] == 'f1-score'
        clf = parts[1]
        assert parts[2] in ['micro', 'macro']
        m = mapping.get(clf)
        idxs[idx] = clf if m is None else m
    df2 = df.rename(index=idxs)
    return df2

def process_file(filename):
    df = pd.read_excel(filename)
    df_micro = process_dataframe(df, ['f1-score', 'micro avg']).transpose()
    df_macro = process_dataframe(df, ['f1-score', 'macro avg']).transpose()
    df_micro = rename_indexes(df_micro)
    df_macro = rename_indexes(df_macro)
    assert len(df_micro.columns) == len(df_macro.columns)
    latex = [
        dataframe_to_latex(df_micro.iloc[:,i:i+1], df_macro.iloc[:,i:i+1])
        for i in range(len(df_micro.columns))
    ]
    return latex

def process_files(filenames):
    latex = [process_file(f) for f in filenames]
    return latex

def write_latex(latex, filename):
    f = open(filename, 'w')
    f.write(latex)
    f.close()

def main():
    if len(argv) < 3:
        print("Usage: %s <output filename> <Excel filename>+" % (argv[0]))
        quit()
    output_filename = argv[1]
    latex = process_files(argv[2:])
    latex = ['\n'.join(l) for l in latex]
    latex = '\n\n'.join(latex)
    write_latex(latex, output_filename)

if __name__ == '__main__':
    main()
