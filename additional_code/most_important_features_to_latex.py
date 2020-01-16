import pandas as pd
from sys import argv
from tqdm import tqdm

def process_file(filename, n_features):
    df = pd.read_excel(filename)
    df = df[0:n_features]
    df = df.drop('Unnamed: 0', 1)
    for col in tqdm(iterable=df.columns, unit='column'):
        df[col][0] = ','.join([str(v) for v in df[col]])
    df = df[0:1]
    df = df.transpose()
    latex = df.to_latex(header=False, index=True)
    return latex

def process_files(filenames, n_features):
    latex = [process_file(f, n_features) for f in tqdm(iterable=filenames, unit='file')]
    return latex

def write_latex(latex, filename):
    f = open(filename, 'w')
    f.write(latex)
    f.close()

def main():
    if len(argv) < 4:
        print('Usage: %s <output filename> <number of features> <Excel filename>+' % (argv[0]))
        quit()
    pd.set_option('display.max_colwidth', -1)
    output_filename = argv[1]
    n_features = int(argv[2])
    latex = process_files(argv[3:], n_features)
    latex = '\n'.join(latex)
    write_latex(latex, output_filename)
    print()

if __name__ == '__main__':
    main()
