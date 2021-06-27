import numpy as np
#import cPickle as pickle
import pickle
from sklearn.decomposition import PCA
import subprocess
import io


n = 10
codes = ["pa"]

for code in codes:
    Glove = {}

    fname = 'cc.{0}.300.vec'.format(code)
    out_file_name = "cc.{0}.{1}.vec".format(code, str(n))
    #f = open('wiki.hi.align.vec')
    f = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')

    import io

    def load_vectors(fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = list(map(float, tokens[1:]))
        return data

    print("Loading vectors.", code)

    Glove = load_vectors(fname=fname)

    print("Done.")
    X_train = []
    X_train_names = []
    for x, vec in Glove.items():
            X_train.append(Glove[x])
            X_train_names.append(x)

    X_train = np.array(X_train)

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

    final_pca_embeddings = {}
    embedding_file = open(out_file_name, 'w')
    embedding_file.write(str(len(X_train_names))+" "+str(n)+"\n")
    for i, x in enumerate(X_train_names[:15]):
        final_pca_embeddings[x] = X_new_final[i]
        embedding_file.write("%s " % x)
        for u in Ufit[0:7]:
            final_pca_embeddings[x] = final_pca_embeddings[x] - np.dot(u.transpose(),final_pca_embeddings[x]) * u 
            
        for t in final_pca_embeddings[x]:
            embedding_file.write("%f " % t)
            
        embedding_file.write("\n")
    
    print(code, " :: ", len(Glove))


