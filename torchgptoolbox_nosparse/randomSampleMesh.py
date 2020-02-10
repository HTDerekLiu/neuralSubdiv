import torch

def randomSampleMesh(V, F, nPt):
    """
    RANDOMSAMPLEMESH randomly samples nPt points on a triangle mesh

    Input:
        V (|V|,3) torch float tensor of vertex positions
        F (|F|,3) torch long tensor of face indices
        nPt number of points to sample
    Output:
        P (nPt,3) torch float tensor of sampled point positions
    """
    nF = F.size()[0]

    FIdx = torch.randint(nF, (nPt,))
    bary = torch.rand(nPt, 3)
    rowSum = torch.sum(bary, 1)
    bary[:,0] /= rowSum
    bary[:,1] /= rowSum
    bary[:,2] /= rowSum

    b0 = bary[:,0:1].repeat(1,3)
    b1 = bary[:,1:2].repeat(1,3)
    b2 = bary[:,2:3].repeat(1,3)

    P = b0*V[F[FIdx,0],:] + b1*V[F[FIdx,1],:] + b2*V[F[FIdx,2],:]
    return P