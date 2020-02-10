import torch 

def setdiff1d(tensor1, tensor2):
    '''
    setdiff1d returns non-intersected elements between tensor1 and tensor2
    '''
    idx = torch.ones_like(tensor1, dtype=torch.bool)
    for ele in tensor2:
        idx = idx & (tensor1 != ele)
    diffEle = tensor1[idx]
    return diffEle