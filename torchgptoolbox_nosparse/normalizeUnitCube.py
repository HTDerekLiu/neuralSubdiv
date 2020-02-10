import torch

def normalizeUnitCube(V):
    '''
    NORMALIZEUNITCUBE normalize a shape to the bounding box by 0.5,0.5,0.5

    Inputs:
        V (|V|,3) torch array of vertex positions

    Outputs:
        V |V|-by-3 torch array of normalized vertex positions
    '''
    V = V - torch.min(V,0)[0].unsqueeze(0)
    V = V / torch.max(V.view(-1)) / 2.0
    return V
