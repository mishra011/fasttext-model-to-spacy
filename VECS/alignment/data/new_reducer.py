import numpy as np
#import cPickle as pickle
import pickle
from sklearn.decomposition import PCA
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
    return words, x



n = 5
codes = ["pa", "pnb"]

for code in codes:
    Glove = {}

    fname = 'cc.{0}.300.vec'.format(code)
    out_file_name = "cc.{0}.{1}.vec".format(code, str(n))

    X_train_names, X_train = load_vectors(fname)
    print("Done.")

    pca_embeddings = {}

    # PCA Dim Reduction
    pca =  PCA(n_components = n)

    X_train = X_train - np.mean(X_train)
    X_new_final = pca.fit_transform(X_train)

    # PCA to do Post-Processing
    pca =  PCA(n_components = n)
    X_new = X_new_final - np.mean(X_new_final)
    X_new = pca.fit_transform(X_new)
    Ufit = pca.components_

    X_new_final = X_new_final - np.mean(X_new_final)

    for i, x in enumerate(X_train_names[:15]):
        tmp = X_new_final[i]
        for u in Ufit[0:7]:
            tmp = tmp - np.dot(u.transpose(),tmp) * u
        
        X_new_final[i] = tmp


    save_vectors(out_file_name,X_new_final, X_train_names)

    print(code , " reduced")


