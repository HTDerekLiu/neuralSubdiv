import os
import sys

cwd = os.getcwd()
sys.path.append(cwd + '/pythongptoolbox/')
sys.path.append(cwd + '/torchgptoolbox_nosparse/')

# torch
import torch
import torchgptoolbox_nosparse as tgp

# pygptoolbox
from readOBJ import readOBJ
from writeOBJ import writeOBJ
from findIdx import findIdx
from midPointUpsampling import midPointUpsampling

# python standard 
import numpy as np
import scipy
import scipy.sparse
import glob
import time
import sys
import json
import pickle as pickle


class Mesh:
    def __init__(self, V, F, hfList):
        # V: vertex list
        # F: face list
        # hfList: half flap list: 
        #   [[vi, vj, flap0, flap1, v_next],
        #   [vj, vi, flap1, flap0, v_next], ....]
        self.V = V
        self.F = F
        self.hfList = hfList


class Category:
    def __init__(self, folder):
        self.meshes = processTrainShapes(folder)
        self.nM = len(self.meshes)
        self.nS = len(self.meshes[0])


class Categories:
    def __init__(self, folders):
        nShape = len(folders)
        self.meshes = []
        for fIdx in range(nShape):
            folder = folders[fIdx]
            S = Category(folder)
            for ii in range(S.nM):
                self.meshes.append(S.meshes[ii])
        self.nM = len(self.meshes)  # number of meshes
        self.nS = len(self.meshes[0])  # number of subdivision levels

        # parameters
        self.hfList = None  # half flap index information
        self.poolMats = None  # vertex one-ring pooling matrices
        self.dofs = None  # vertex degrees of freedom
        self.LCs = None  # vector of differential coordinates

    def getTrainData(self, mIdx, params):
        input = torch.cat((
            self.meshes[mIdx][0].V,
            self.LCs[mIdx]),
            dim=1)
        return input  # (nV x Din)

    def getHalfFlap(self):
        # create a list of tuples
        # HF[meshIdx][subdIdx] = [vi[0], vj[0], flap0, flap1,]
        # HF[meshIdx][subdIdx] = [vj[0], vi[0], flap1, flap0,]
        HF = [None] * self.nM
        for ii in range(self.nM):
            fifj = [None] * (self.nS - 1)
            for jj in range(self.nS - 1):
                idx = self.meshes[ii][jj].hfList[:, [0, 1, 2, 3]]
                fifj[jj] = idx.reshape(-1, 4)
            HF[ii] = list(fifj)
        return HF

    def getFlapPool(self, HF):
        nM = len(HF)
        nS = len(HF[0])

        poolFlap = [None] * nM
        dof = [None] * nM
        for ii in range(nM):
            poolFlap_ij = [None] * nS
            dof_ij = [None] * nS
            for jj in range(nS):
                hfIdx = HF[ii][jj]
                nV = hfIdx[:, 0].max() + 1

                rIdx = hfIdx[:, 0]
                cIdx = torch.arange(hfIdx.size(0))
                I = torch.cat([rIdx, cIdx], 0).reshape(2, -1)
                val = torch.ones(hfIdx.size(0))
                poolMat = torch.sparse.FloatTensor(I, val, torch.Size([nV, hfIdx.size(0)]))

                rowSum = torch.sparse.sum(poolMat, dim=1).to_dense()

                poolFlap_ij[jj] = poolMat
                dof_ij[jj] = rowSum
            poolFlap[ii] = list(poolFlap_ij)
            dof[ii] = list(dof_ij)
        return poolFlap, dof

    def getLaplaceCoordinate(self, hfList, poolMats, dofs):
        LC = [None] * self.nM
        for mIdx in range(self.nM):
            V = self.meshes[mIdx][0].V
            HF = hfList[mIdx][0]
            poolMat = poolMats[mIdx][0]
            dof = dofs[mIdx][0]
            dV_he = V[HF[:, 0], :] - V[HF[:, 1], :]
            dV_v = torch.spmm(poolMat, dV_he)
            dV_v /= dof.unsqueeze(1)
            LC[mIdx] = dV_v
        return LC

    def computeParameters(self):
        self.hfList = self.getHalfFlap()
        self.poolMats, self.dofs = self.getFlapPool(self.hfList)
        self.LCs = self.getLaplaceCoordinate(self.hfList, self.poolMats, self.dofs)

    def toDevice(self, device):
        for ii in range(self.nM):
            for jj in range(self.nS):
                self.meshes[ii][jj].V = self.meshes[ii][jj].V.to(device)
                self.meshes[ii][jj].F = self.meshes[ii][jj].F.to(device)
        for ii in range(self.nM):
            self.LCs[ii] = self.LCs[ii].to(device)
            for jj in range(self.nS - 1):
                self.hfList[ii][jj] = self.hfList[ii][jj].to(device)
                self.poolMats[ii][jj] = self.poolMats[ii][jj].to(device)
                self.dofs[ii][jj] = self.dofs[ii][jj].to(device)


class TestCategories:
    def __init__(self, meshPathList, nSubd=2):
        nShape = len(meshPathList)
        self.meshes = preprocessTestShapes(meshPathList, nSubd)
        self.nM = len(self.meshes)  # number of meshes
        self.nS = len(self.meshes[0])  # number of subdivision levels

        # parameters
        self.hfList = None  # half flap index information
        self.poolMats = None  # vertex one-ring pooling matrices
        self.dofs = None  # vertex degrees of freedom
        self.LCs = None  # vector of differential coordinates

    def getTrainData(self, mIdx, params):
        input = torch.cat((
            self.meshes[mIdx][0].V,
            self.LCs[mIdx]),
            dim=1)
        return input  # (nV x Din)

    def getHalfFlap(self):
        # create a list of tuples
        # HF[meshIdx][subdIdx] = [vi[0], vj[0], flap0, flap1,]
        # HF[meshIdx][subdIdx] = [vj[0], vi[0], flap1, flap0,]
        HF = [None] * self.nM
        for ii in range(self.nM):
            fifj = [None] * (self.nS - 1)
            for jj in range(self.nS - 1):
                idx = self.meshes[ii][jj].hfList[:, [0, 1, 2, 3]]
                fifj[jj] = idx.reshape(-1, 4)
            HF[ii] = list(fifj)
        return HF

    def getFlapPool(self, HF):
        nM = len(HF)
        nS = len(HF[0])

        poolFlap = [None] * nM
        dof = [None] * nM
        for ii in range(nM):
            poolFlap_ij = [None] * nS
            dof_ij = [None] * nS
            for jj in range(nS):
                hfIdx = HF[ii][jj]
                nV = hfIdx[:, 0].max() + 1

                rIdx = hfIdx[:, 0]
                cIdx = torch.arange(hfIdx.size(0))
                I = torch.cat([rIdx, cIdx], 0).reshape(2, -1)
                val = torch.ones(hfIdx.size(0))
                poolMat = torch.sparse.FloatTensor(I, val, torch.Size([nV, hfIdx.size(0)]))

                rowSum = torch.sparse.sum(poolMat, dim=1).to_dense()

                poolFlap_ij[jj] = poolMat
                dof_ij[jj] = rowSum
            poolFlap[ii] = list(poolFlap_ij)
            dof[ii] = list(dof_ij)
        return poolFlap, dof

    def getLaplaceCoordinate(self, hfList, poolMats, dofs):
        LC = [None] * self.nM
        for mIdx in range(self.nM):
            V = self.meshes[mIdx][0].V
            HF = hfList[mIdx][0]
            poolMat = poolMats[mIdx][0]
            dof = dofs[mIdx][0]
            dV_he = V[HF[:, 0], :] - V[HF[:, 1], :]
            dV_v = torch.spmm(poolMat, dV_he)
            dV_v /= dof.unsqueeze(1)
            LC[mIdx] = dV_v
        return LC

    def computeParameters(self):
        self.hfList = self.getHalfFlap()
        self.poolMats, self.dofs = self.getFlapPool(self.hfList)
        self.LCs = self.getLaplaceCoordinate(self.hfList, self.poolMats, self.dofs)

    def toDevice(self, device):
        for ii in range(self.nM):
            for jj in range(self.nS):
                if jj == 0:
                    self.meshes[ii][jj].V = self.meshes[ii][jj].V.to(device)
                self.meshes[ii][jj].F = self.meshes[ii][jj].F.to(device)
        for ii in range(self.nM):
            self.LCs[ii] = self.LCs[ii].to(device)
            for jj in range(self.nS - 1):
                self.hfList[ii][jj] = self.hfList[ii][jj].to(device)
                self.poolMats[ii][jj] = self.poolMats[ii][jj].to(device)
                self.dofs[ii][jj] = self.dofs[ii][jj].to(device)


def computeFlapList(V, F, numSubd=2):
    FList = []
    halfFlapList = []

    for iter in range(numSubd):
        # compute the subdivision indices
        nV = V.size(0)
        VV, FF, S = tgp_midPointUp(V, F, 1)
        rIdx = S._indices()[0, :]
        cIdx = S._indices()[1, :]
        val = S._values()

        # only extract new vertices
        cIdx = cIdx[rIdx >= nV]
        val = val[rIdx >= nV]
        rIdx = rIdx[rIdx >= nV]
        assert ((val == 0.5).all())

        rIdx, idx = torch.sort(rIdx + nV)
        cIdx = cIdx[idx]
        rIdx = rIdx[::2]  # same as MATLAB rIdx = rIdx[1:2:end]
        cIdx = cIdx.view(-1, 2)
        # Note: Vodd = (V[cIdx[:,0],:] + V[cIdx[:,1],:]) / 2.0

        flapIdx = torch.zeros(len(rIdx), 5).long()
        for kk in range(len(rIdx)):
            vi = cIdx[kk, 0]
            vj = cIdx[kk, 1]
            adjFi, _ = tgp.findIdx(F, vi)
            adjFj, _ = tgp.findIdx(F, vj)
            adjF = tgp.intersect1d(adjFi, adjFj)
            assert (adjF.size(0) == 2)

            f1 = F[adjF[0], :]
            f2 = F[adjF[1], :]

            # roll the index so that f1_vi[0] == f2_vi[0] == vi
            f1roll = -tgp.findIdx(f1, vi)[0]
            f2roll = -tgp.findIdx(f2, vi)[0]
            f1_vi = tgp.roll1d(f1, f1roll)
            f2_vi = tgp.roll1d(f2, f2roll)
            assert (f1_vi[0] == vi)
            assert (f2_vi[0] == vi)

            # check which one is f = [vi, vj, ?]
            if f1_vi[1] == vj:
                f_first = f1_vi  # f_first = [vi, vj, ?]
                f_sec = f2_vi  # f_sec   = [vi, ?, vj]
            elif f2_vi[1] == vj:
                f_first = f2_vi  # f_first = [vi, vj, ?]
                f_sec = f1_vi  # f_sec   = [vi, ?, vj]
            assert (f_first[1] == vj)
            assert (f_sec[2] == vj)

            # assemble flapIdx = [vi vj flap1, flap2, v]
            flapIdx[kk, :] = torch.tensor([vi, vj, f_first[2], f_sec[1], rIdx[kk]])

        # compute half flap list
        # note:
        # halfFlapIdx = [[vi, vj, vk, vl, idxInVV]
        #                [vj, vi, vl, vk, idxInVV]]
        halfFlapIdx = flapIdx[:, [0, 1, 2, 3, 4, 1, 0, 3, 2, 4]]
        halfFlapIdx = halfFlapIdx.reshape(-1, 5)

        FList.append(F)
        halfFlapList.append(halfFlapIdx)

        V = VV
        F = FF

    FList.append(F)
    halfFlapList.append(None)
    return FList, halfFlapList


def tgp_midPointUp(V, F, subdIter=1):
    Vnp = V.data.numpy()
    Fnp = F.data.numpy()
    VVnp, FFnp, SSnp = midPointUpsampling(Vnp, Fnp, subdIter)
    VV = torch.from_numpy(VVnp).float()
    FF = torch.from_numpy(FFnp).long()

    SSnp = SSnp.tocoo()
    values = SSnp.data
    indices = np.vstack((SSnp.row, SSnp.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = SSnp.shape
    SS = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return VV, FF, SS


def processTrainShapes(folder):
    # detect number of subd and number of meshes per subd
    subFolders = os.listdir(folder)
    subFolders = [x for x in subFolders if not (x.startswith('.'))]
    nSubd = len(subFolders) - 1

    objPaths = glob.glob(folder + subFolders[0] + "/*.obj")
    nObjs = len(objPaths)

    # create data loading list of strings (glod results are not sorted)
    paths = []
    for ii in range(nSubd + 1):
        paths.append(folder + 'subd' + str(ii) + '/')

    objFiles = []
    for ii in range(nObjs):
        objFiles.append(str(ii + 1).zfill(3))

    # load data
    meshes = [None] * nObjs
    for ii in range(nObjs):
        print('process meshes %d / %d' % (ii, nObjs))
        meshes_i = [None] * (nSubd + 1)
        for jj in range(nSubd + 1):
            V, F = tgp.readOBJ(paths[jj] + objFiles[ii] + '.obj')
            _, hfList = computeFlapList(V, F, 1)
            meshes_i[jj] = Mesh(V, F, hfList[0])
        meshes[ii] = list(meshes_i)
    print('num Subd: %d, num meshes: %d' % (nSubd, nObjs))
    return meshes


def random3DRotation():
    theta_x = torch.rand(1) * 2 * np.pi
    sinx = torch.sin(theta_x)
    cosx = torch.cos(theta_x)
    Rx = torch.tensor([[1., 0., 0.],
                       [0., cosx, -sinx],
                       [0., sinx, cosx]])

    theta_y = torch.rand(1) * 2 * np.pi
    siny = torch.sin(theta_y)
    cosy = torch.cos(theta_y)
    Ry = torch.tensor([[cosy, 0., siny],
                       [0., 1., 0.],
                       [-siny, 0., cosy]])

    theta_z = torch.rand(1) * 2 * np.pi
    sinz = torch.sin(theta_z)
    cosz = torch.cos(theta_z)
    Rz = torch.tensor([[cosz, -sinz, 0.],
                       [sinz, cosz, 0.],
                       [0., 0., 1.]])
    return Rx.mm(Ry).mm(Rz)


def preprocessTestShapes(meshPathList, nSubd=2):
    nObjs = len(meshPathList)
    meshes = [None] * nObjs
    for meshIdx in range(nObjs):
        path = meshPathList[meshIdx]
        V, F = tgp.readOBJ(path)
        V = tgp.normalizeUnitCube(V) * 2
        FList, hfList = computeFlapList(V, F, nSubd)

        meshes_i = [None] * (nSubd + 1)
        for jj in range(nSubd + 1):
            if jj == 0:
                meshes_i[jj] = Mesh(V, FList[jj], hfList[jj])
            else:
                meshes_i[jj] = Mesh(None, FList[jj], hfList[jj])
        meshes[meshIdx] = list(meshes_i)
    print('num Subd: %d, num meshes: %d' % (nSubd, nObjs))
    return meshes
