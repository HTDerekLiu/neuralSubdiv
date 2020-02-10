from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from include import *
from models import *
NETPARAMS = 'netparams.dat'

# should run the file like "python train.py ./path/to/folder/"
def main():

    # load hyper parameters
    folder = sys.argv[1]
    with open(folder + 'hyperparameters.json', 'r') as f:
        params = json.load(f)

    # load traininig data
    S = pickle.load(open(params["train_pkl"], "rb"))
    S.computeParameters()
    S.toDevice(params["device"])
    
    # load validation set
    T = pickle.load(open(params["valid_pkl"], "rb"))    
    T.computeParameters()
    T.toDevice(params["device"])

    # initialize network 
    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
    net = SubdNet(params)
    net = net.to(params['device'])
    net.apply(init_weights)

    # loss function
    lossFunc = torch.nn.MSELoss().to(params['device'])

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(),lr=params['lr'])

    # training
    trainLossHis = []
    validLossHis = []
    bestLoss = np.inf
    nSub = params["numSubd"]
    for epoch in range(params['epochs']):      
        ts = time.time()

        # loop over training shapes 
        trainErr = 0.0
        for mIdx in range(S.nM):
            # forward pass
            x = S.getTrainData(mIdx, params)
            outputs = net(x,mIdx, S.hfList, S.poolMats, S.dofs)
            
            # target mesh
            Vt = S.meshes[mIdx][params["numSubd"]].V.to(params['device'])
            Ft = S.meshes[mIdx][params["numSubd"]].F

            # compute loss function
            loss = 0.0
            for ii in range(params["numSubd"]+1):
                nV = outputs[ii].size(0)
                loss += lossFunc(outputs[ii], Vt[:nV,:])

            # move
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record training error
            trainErr += loss.cpu().data.numpy()
        trainLossHis.append(trainErr / S.nM)

        # loop over validation shapes 
        validErr = 0.0
        for mIdx in range(T.nM):
            x = T.getTrainData(mIdx, params)
            outputs = net(x,mIdx,T.hfList,T.poolMats,T.dofs)
            
            # target mesh
            Vt = T.meshes[mIdx][params["numSubd"]].V.to(params['device'])
            Ft = T.meshes[mIdx][params["numSubd"]].F

            # compute loss function
            loss = 0.0
            for ii in range(params["numSubd"]+1):
                nV = outputs[ii].size(0)
                loss += lossFunc(outputs[ii], Vt[:nV,:])
                
            # record validation error
            validErr += loss.cpu().data.numpy()
        validLossHis.append(validErr / T.nM)

        # save the best model
        if validErr < bestLoss:
            bestLoss = validErr
            torch.save(net.state_dict(),params['output_path'] + NETPARAMS)

        print("epoch %d, train loss %.6e, valid loss %.6e, remain time: %s" % (epoch, trainLossHis[-1], validLossHis[-1], int(round((params['epochs']-epoch) * (time.time() - ts)))))

    # save loss history
    np.savetxt(params['output_path']+'train_loss.txt', np.array(trainLossHis), delimiter=',')
    np.savetxt(params['output_path']+'valid_loss.txt', np.array(validLossHis), delimiter=',')

    # write output shapes (validation set)
    mIdx = 0
    x = T.getTrainData(mIdx, params)
    outputs = net(x, mIdx,T.hfList,T.poolMats,T.dofs) 

    # write unrotated outputs
    tgp.writeOBJ(params['output_path'] + str(mIdx) +'_oracle.obj',\
        T.meshes[mIdx][len(outputs)-1].V.to('cpu'),\
        T.meshes[mIdx][len(outputs)-1].F.to('cpu'))
    for ii in range(len(outputs)):
        x = outputs[ii].cpu()
        tgp.writeOBJ(params['output_path'] + str(mIdx) +'_subd' + str(ii) + '.obj',x, T.meshes[mIdx][ii].F.to('cpu'))

    # write rotated output shapes (validation set)
    mIdx = 0
    x = T.getTrainData(mIdx, params)
    outputs = net(x, mIdx,T.hfList,T.poolMats,T.dofs)

    dV = torch.rand(1, 3).to(params['device'])
    R = random3DRotation().to(params['device'])
    x[:,:3] = x[:,:3].mm(R.t())
    x[:,3:] = x[:,3:].mm(R.t())
    x[:,:3] += dV
    outputs = net(x, mIdx,T.hfList,T.poolMats,T.dofs) 

    # write rotated outputs
    for ii in range(len(outputs)):
        x = outputs[ii].cpu()
        tgp.writeOBJ(params['output_path'] + str(mIdx) +'_rot_subd' + str(ii) + '.obj',x, T.meshes[mIdx][ii].F.to('cpu'))
        
if __name__ == '__main__':
    main()