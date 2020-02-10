import torch 

def roll1d(x, n):  
    return torch.cat((x[-n:], x[:-n]))