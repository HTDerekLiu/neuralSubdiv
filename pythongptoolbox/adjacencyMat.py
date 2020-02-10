import numpy as np
import scipy
import scipy.sparse

def adjacencyMat(F):
    '''
    adjacencyMat computes vertex adjacency matrix

    Inputs:
    F: |F|-by-3 numpy ndarray of face indices

    Outputs:
    A: sparse.csr_matrix with size |V|-by-|V|
    '''

    idx = np.array([[0,1], [1,2], [2,0]]) # assume we have simplex with DOF=3
    edgeIdx1 = np.reshape(F[:,idx[:,0]], (np.product(F.shape)))
    edgeIdx2 = np.reshape(F[:,idx[:,1]], (np.product(F.shape)))
    data = np.ones([len(edgeIdx1)])
    numVert = np.amax(F)+1
    A = scipy.sparse.csr_matrix((data, (edgeIdx1, edgeIdx2)), shape=(numVert,numVert), dtype = np.int32)
    return A