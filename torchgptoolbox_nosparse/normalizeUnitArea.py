import torch
from . faceAreas import faceAreas

def normalizeUnitArea(V,F):
    '''
    NORMALIZEUNITAREA normalize a shape to have total surface area 1

    Inputs:
        V (|V|,3) torch array of vertex positions
        F (|F|,3) torch array of face indices

    Outputs:
        V |V|-by-3 torch array of normalized vertex positions
    '''
    totalArea = torch.sum(faceAreas(V,F))
    V = V / torch.sqrt(totalArea)
    return V
