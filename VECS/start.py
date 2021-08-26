import fasttext

fname = "/home/deepak/VECS/wiki.hi.align.vec"
#model = fasttext.load_model()


import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


# data = load_vectors(fname=fname)

# print(data)

model = fasttext.load_model(fname)
