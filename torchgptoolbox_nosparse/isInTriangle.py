import torch
from . rowwiseDet2D import rowwiseDet2D

def isInTriangle(P,UV,F):
    # INOUTTRI detect inside(True)/outside(False) of a query point P from 
    # a triangle mesh in R2 (V,F)
    
    # Inputs:
    #     P:  length 2 vector
    #     UV: nUV x 2 vertices in 2D
    #     F:  nF x 3 faces
       
    # Outputs:
    #     isInF: nF x 1 bool whether P is in triangle Fi

    nUV = UV.size(0)
    P1 = UV[F[:,0],:]
    P2 = UV[F[:,1],:]
    P3 = UV[F[:,2],:]
    P12 = P1 - P2
    P23 = P2 - P3
    P31 = P3 - P1
    
    detP31P23 = rowwiseDet2D(P31,  P23)
    detP30P23 = rowwiseDet2D(P3-P, P23)
    detP12P31 = rowwiseDet2D(P12,  P31)
    detP10P31 = rowwiseDet2D(P1-P, P31)
    detP23P12 = rowwiseDet2D(P23,  P12)
    detP20P12 = rowwiseDet2D(P2-P, P12)

    isInF = (detP31P23 * detP30P23) >= -1e-6
    isInF = isInF & ((detP12P31 * detP10P31) >= -1e-6)
    isInF = isInF & ((detP23P12 * detP20P12) >= -1e-6)
    
    return isInF