import torch
from . plotMesh import plotMesh
import matplotlib.pyplot as plt


def computeBarycentric2D(p, UV, F):
    '''
    computeBarycentric2D computes berycentric coordinate or queryUV in fUV
    Inputs:
        p  length 2 array
        UV |UV| by 2 array
        F  |F| by 3 array
    Outputs:
        B  |F| by 3 array of barycentric coordinate from p to all F
    '''

    a = UV[F[:,0],:]
    b = UV[F[:,1],:]
    c = UV[F[:,2],:]

    nF = F.size(0)
    v0 = b-a
    v1 = c-a
    v2 = p.unsqueeze(0)-a


    d00 = v0.mul(v0).sum(1)
    d01 = v0.mul(v1).sum(1)
    d11 = v1.mul(v1).sum(1)
    d20 = v2.mul(v0).sum(1)
    d21 = v2.mul(v1).sum(1)
    denom = d00.mul(d11) - d01.mul(d01)
    denom = 1.0 / denom

    v = (d11.mul(d20) - d01.mul(d21)).mul(denom)
    w = (d00.mul(d21) - d01.mul(d20)).mul(denom)
    u = 1 - v - w

    B = torch.cat((u.unsqueeze(1),v.unsqueeze(1),w.unsqueeze(1)), dim = 1)
    return B


    # iUV = fUV[0,:]
    # jUV = fUV[1,:]
    # kUV = fUV[2,:]

    # # compute barycentric coordinate
    # v0 = jUV - iUV
    # v1 = kUV - iUV
    # v2 = queryUV - iUV
    # d00 = v0.dot(v0) 
    # d01 = v0.dot(v1)
    # d11 = v1.dot(v1)
    # d20 = v2.dot(v0)
    # d21 = v2.dot(v1)
    # denom = d00*d11 - d01*d01
    # bj = (d11 * d20 - d01 * d21) / denom
    # bk = (d00 * d21 - d01 * d20) / denom
    # bi = 1 - bj - bk

    # # check solution
    # reconUV = bi*iUV + bj*jUV + bk*kUV
    # assert(torch.norm(reconUV-queryUV) < 1e-6)

    # # filter 
    # b = torch.tensor([bi,bj,bk])
    # # assert((b >= -1e-5).all())
    # if ~(b >= -1e-5).all():
    #     print(b)
    #     print(queryUV)
    #     print(fUV)
    #     ax = plotMesh(fUV,torch.tensor([[0,1,2]]),showEdges= True)
    #     P = queryUV.data.numpy()
    #     ax.scatter(P[0],P[1],0,c='r')
    #     plt.show()
    
    # assert((b <= 1.).all())
    # assert torch.abs(torch.sum(b) - 1.0) < 1e-5
    # b = b + 1e-6
    # b = b / torch.sum(b)

    # return b


