from task import clustering

import numpy as np

if __name__ == '__main__':
    emb = np.load('emb.npy', allow_pickle=True)
    nmi, ari = clustering(emb)
    print(nmi, ari)