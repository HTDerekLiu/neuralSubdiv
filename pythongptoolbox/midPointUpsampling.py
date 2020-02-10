import numpy as np
import scipy
from adjacencyMat import adjacencyMat

def midPointUpsampling(V,F,numIter=1):
    '''
    midPointUpsampling do mid point upsampling 

    Inputs:
        V (|V|,3) numpy array of vertex positions
        F (|F|,3) numpy array of face indices
        numIter number of upsampling to perform

    Outputs:
        V |V|-by-3 numpy array of new vertex positions
        F |F|-by-3 numpy array of new face indices
        upOpt |Vup|-by-|V| numpy array of upsampling operator

    TODO:
        add boundary constraints 
    '''
    for iter in range(numIter):
        nV = V.shape[0]
        nF = F.shape[0]

        # compute new vertex positions
        hE = np.concatenate( (F[:,[0,1]], F[:,[1,2]], F[:,[2,0]]), axis=0 )
        hE = np.sort(hE, axis = 1)
        E, hE2E = np.unique(hE, axis=0, return_inverse=True)
        nE = E.shape[0]
        newV = (V[E[:,0],:] + V[E[:,1],:]) / 2.0
        V = np.concatenate( (V, newV), axis = 0 )

        # compute updated connectivity
        i2 = nV       + np.arange(nF)
        i0 = nV+nF    + np.arange(nF)
        i1 = nV+nF+nF + np.arange(nF)

        hEF0 = np.concatenate( (F[:,0:1], i2[:,None], i1[:,None]), axis=1 )
        hEF1 = np.concatenate( (F[:,1:2], i0[:,None], i2[:,None]), axis=1 )
        hEF2 = np.concatenate( (F[:,2:3], i1[:,None], i0[:,None]), axis=1 )
        hEF3 = np.concatenate( (i0[:,None], i1[:,None], i2[:,None]), axis=1 )
        hEF  = np.concatenate( (hEF0, hEF1, hEF2, hEF3), axis=0 )

        hE2E =  np.concatenate( (np.arange(nV), hE2E+nV), axis=0 )
        uniqV = np.unique(F)
        F = hE2E[hEF]

        # upsampling for odd vertices
        rIdx = uniqV
        cIdx = uniqV
        val = np.ones((len(uniqV),))

        # upsampling for even vertices
        rIdx = np.concatenate( (rIdx, nV+np.arange(nE),  nV+np.arange(nE)) )
        cIdx = np.concatenate( (cIdx, E[:,0],  E[:,1]) )
        val = np.concatenate( (val, np.ones(2*nE)*0.5) )

        # upsampling operator
        if iter == 0:
            S = scipy.sparse.coo_matrix( (val, (rIdx,cIdx)), shape = (nV+nE, nV) )
        else:
            tmp = scipy.sparse.coo_matrix( (val, (rIdx,cIdx)), shape = (nV+nE, nV) )
            S = tmp * S

    return V, F, S