import torch
def findIdx(F, VIdx):
    '''
    FINDIDX finds desired indices in a torch tensor

    Inputs:
    F: |F|-by-dim torch tensor 
    VIdx: a list of indices

    Output:
    r, c: row/colummn indices in the torch tensor
    '''

    def isin(ar1, ar2):
        return (ar1[..., None] == ar2).any(-1)

    mask = isin(F.view(-1),VIdx)
    try:
        nDim = F.shape[1]
    except:
        nDim = 1
    r = torch.floor(torch.where(mask)[0] / (nDim*1.0) ).long()
    c = torch.where(mask)[0] % nDim
    return r,c