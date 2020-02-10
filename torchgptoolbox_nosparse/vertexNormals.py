import torch 
import sys
from . normalizeRow import normalizeRow

def vertexNormals(V,F):
    vec1 = V[F[:,1],:] - V[F[:,0],:]
    vec2 = V[F[:,2],:] - V[F[:,0],:]
    FN = torch.cross(vec1, vec2) / 2

    rIdx = F.view(-1)
    cIdx = torch.arange(F.size(0))
    cIdx = cIdx.unsqueeze(1).repeat(1,3).view(-1)
    val = torch.ones(cIdx.size(0))

    I = torch.cat([rIdx,cIdx], 0).reshape(2, -1)
    W = torch.sparse.FloatTensor(I, val, torch.Size([V.size(0),F.size(0)]))
    VN = torch.sparse.mm(W, FN)
    VN = normalizeRow(VN)
    return VN