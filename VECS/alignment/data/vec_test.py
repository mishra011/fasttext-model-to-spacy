

import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


data_pa = load_vectors(fname="cc.pa.300.vec")

data_pnb = load_vectors(fname="cc.pnb.300.vec")


print(len(data_pa), "pa")
print(len(data_pnb), "pnb")


k = list(set(list(data_pa.keys())).intersection(set(data_pnb.keys())))
print(len(k))

print(k[:300])