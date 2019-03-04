import pickle

def dump(obj, filename):
    output_file = open(filename, 'wb')
    pickle.dump(obj=obj, file=output_file, protocol=2)
    output_file.close()

def load(filename):
    input_file = open(filename, 'rb')
    data = pickle.load(file=input_file)
    input_file.close()
    return data
