import torch

def rowwiseDet2D(v1List,v2List):
    '''
    rowwiseDet2D computes the determinant between two sets of 2D vectors.
    This is equivalent of 

    for ii in range(v1List.size(0)):
        v1 = v1List[ii,:];
        v2 = v2List[ii,:];
        detList[ii] = det(v1,v2);

    Inputs:
      v1List nV x 2 matrix
      v2List nV x 2 matrix

    Outputs:
      detList nV x 1 determinant
    '''
    assert(v1List.size(1) == 2)
    assert(v2List.size(1) == 2)
    assert(v1List.size(0) == v2List.size(0))

    nV = v1List.size(0)
    M = torch.zeros((2,2,nV))
    M[0,:,:] = v1List.t()
    M[1,:,:] = v2List.t()
    Mvec = M.view(2*2,nV)
    detList = Mvec[0,:] * Mvec[3,:] - Mvec[1,:] * Mvec[2,:]
    return detList