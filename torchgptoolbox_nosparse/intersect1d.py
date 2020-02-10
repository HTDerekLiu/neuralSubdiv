import torch 

def intersect1d(tensor1, tensor2):
    '''
    intersect1d return intersected elements between tensor1 and tensor2
    '''
    aux = torch.cat((tensor1, tensor2),dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]