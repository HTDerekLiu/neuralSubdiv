clc; clear all; close all;
addpath('./utils_matlab/')
% this depends on gptoolbox: https://github.com/alecjacobson/gptoolbox

% read the input mesh
meshPath = './data_meshes/objs_original/bob.obj';
numSubd = 2;
numData = 1;
nV_variance = 100;
nV_average = 300;

% extract mesh name
[~, meshName, ~] = fileparts(meshPath);

% create folders to store meshes
folderList = {};
for ii = 0:numSubd
    folder = strcat('./data_meshes/',meshName, "_", num2str(numData),'/subd',num2str(ii),'/');
    mkdir(folder)
    folderList{ii+1} = folder;
end

% load mesh and normalize 
[Vin,Fin] = load_mesh(meshPath);
Vin = normalizeUnitBox(Vin);

% create subdivision meshes
for ii = 1:numData
    nVc = nV_average + round(nV_variance * (rand() - 0.5));

    % perform qslim decimation with out self-parameterization (Section 4.1,4.2)
    [V,F,decInfo] = SSP_decimation(Vin,Fin,nVc, 'QEM', 0.1,0.4,true); 

    % upsampling with subdivision connectivity & use self-parametrization to query ground truth position (Fig.13)
    [VList,FList] = SSP_upsample(V,F,numSubd,Vin,Fin,decInfo);

    % clean all the upsampled/ground truth meshes (Fig.8 all the meshes in single column) because they have some redundant vertices
    [RV,IM,J] = remove_unreferenced(VList{1},FList{1});
    RF = IM(FList{1});
    subd(1).V = RV;
    subd(1).F = RF;
    [~,IM,~] = remove_unreferenced(VList{end},FList{end});
    for jj = 2:length(VList)
        [RV,IMtmp,J] = remove_unreferenced(VList{jj},FList{jj});
        RF = IMtmp(FList{jj});
        subd(jj).V = RV;
        subd(jj).F = RF;
    end

    % write the output meshes
    meshIdxStr = num2str(ii,'%03.f');
    writeOBJ(strcat(folderList{1},meshIdxStr,'.obj'), subd(1).V, subd(1).F);
    for jj = 2:length(VList)
        writeOBJ(strcat(folderList{jj},meshIdxStr,'.obj'), subd(jj).V, subd(jj).F);
    end
end
