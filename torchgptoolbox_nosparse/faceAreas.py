import torch

def faceAreas(V, F):
    """
    FACEAREAS computes area per face 

    Input:
        V (|V|,3) torch float tensor of vertex positions
        F (|F|,3) torch long tensor of face indices
    Output:
        FA (|F|,) torch tensor of face area
    """
    vec1 = V[F[:,1],:] - V[F[:,0],:]
    vec2 = V[F[:,2],:] - V[F[:,0],:]
    FN = torch.cross(vec1, vec2) / 2
    FA = torch.sqrt(torch.sum(FN.pow(2),1))
    return FA