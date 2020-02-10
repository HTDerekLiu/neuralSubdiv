import torch

def normalizeRow(X):
    """
    NORMALIZEROW normalizes the l2-norm of each row in a np array 

    Input:
        X: n-by-m torch tensor
    Output:
        X_normalized: n-by-m row normalized torch tensor
    """
    l2norm = torch.sqrt(torch.sum(X.pow(2),1))
    X = X / l2norm.unsqueeze(1)
    # nCol = X.size()[1]
    # for cIdx in range(nCol):
    #     X[:,cIdx] /= l2norm
    return X