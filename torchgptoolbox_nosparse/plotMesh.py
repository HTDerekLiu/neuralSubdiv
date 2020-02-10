import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import numpy as np
import torch

from . faceNormals import faceNormals

def plotMesh(Vin,F, \
    showEdges=False,\
    alpha=0.5):
    """
    PLOTMESH plot a triangle mesh

    Input:
      V (|V|,3) torch tensor of vertex positions
	  F (|F|,3) torch tensor of face indices
    Output:
      None
    """
    V = Vin.clone()
    if V.size(1) == 2:
        V = torch.cat((V, torch.zeros((V.size(0),1))), dim=1)
    FN = faceNormals(V,F)
    V = V.data.numpy()
    F = F.data.numpy()
    FN = FN.data.numpy()

    # compute colors for rendering
    
    z = (FN[:,2] + 3) / 5
    face_color = np.array([144.0/ 255.0, 210.0/ 255.0, 236.0/ 255.0])
    face_color = z[:,None]*face_color

    # open a figure
    fig = plt.figure(figsize=(7, 7))
    ax = fig.gca(projection='3d')

    # get vertices 
    vtx = V[F,:]
    
    # plot
    if showEdges == True:
        mesh = a3.art3d.Poly3DCollection(vtx, linewidths=.5,  edgecolors=[0,0,0], alpha=alpha)
    else:
        mesh = a3.art3d.Poly3DCollection(vtx, alpha=alpha)
    
    # add face color
    mesh.set_facecolor(face_color)
    
    # add mesh to figures
    ax.add_collection3d(mesh)

    # set figure axis 
    actV = np.unique(F.flatten())
    axisRange = np.array([np.max(V[actV,0])-np.min(V[actV,0]), np.max(V[actV,1])-np.min(V[actV,1]), np.max(V[actV,2])-np.min(V[actV,2])])
    r = np.max(axisRange) / 2.0
    mean = np.mean(V[actV,:], 0)
    ax.set_xlim(mean[0]-r, mean[0]+r)
    ax.set_ylim(mean[1]-r, mean[1]+r)
    ax.set_zlim(mean[2]-r, mean[2]+r)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return ax
