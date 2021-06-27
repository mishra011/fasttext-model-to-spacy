

import numpy as np
import pickle
import subprocess
import io
import numpy as np
import collections

def save_vectors(fname, x, words):
    n, d = x.shape
    fout = io.open(fname, 'w', encoding='utf-8')
    fout.write(u"%d %d\n" % (n, d))
    for i in range(n):
        fout.write(words[i] + " " + " ".join(map(lambda a: "%.4f" % a, x[i, :])) + "\n")
    fout.close()


def load_vectors(fname, maxload=20000000, norm=True, center=False, verbose=True):
    if verbose:
        print("Loading vectors from %s" % fname)
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    if maxload > 0:
        n = min(n, maxload)
    x = np.zeros([n, d])
    words = []
    for i, line in enumerate(fin):
        if i >= n:
            break
        tokens = line.rstrip().split(' ')
        words.append(tokens[0])
        v = np.array(tokens[1:], dtype=float)
        x[i, :] = v
    if norm:
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    if center:
        x -= x.mean(axis=0)[np.newaxis, :]
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    if verbose:
        print("%d word vectors loaded" % (len(words)))
    return words, list(x)


base = "en"

codes = ['hi', "mr"]

n = 20

base_model = 'align.{0}.{1}.vec'.format(base, str(n))

words, vecs = load_vectors(base_model)

for code in codes:
    print(code, "=======")
    fname = 'align.{0}.{1}.vec'.format(code, str(n))
    _words, _vecs = load_vectors(fname)

    delta = list(set(_words)- set(words))
    print(len(delta))



    for i,w in enumerate(delta):
        words.append(w)
        k = _words.index(w)
        vecs.append(_vecs[k])
        



save_vectors("combined_mr_hi_20.vec", np.array(vecs), words)

