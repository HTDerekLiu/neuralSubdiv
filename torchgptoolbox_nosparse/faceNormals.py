import torch

def faceNormals(V, F):
    """
    FACENORMALS computes unit normals per face 

    Input:
        V (|V|,3) torch float tensor of vertex positions
        F (|F|,3) torch long tensor of face indices
    Output:
        FN (|F|,3) torch tensor of face normals
    """
    vec1 = V[F[:,1],:] - V[F[:,0],:]
    vec2 = V[F[:,2],:] - V[F[:,0],:]
    FN = torch.cross(vec1, vec2) / 2
    l2norm = torch.sqrt(torch.sum(FN.pow(2),1))
    nCol = FN.size()[1]
    for cIdx in range(nCol):
        FN[:,cIdx] /= l2norm
    return FN