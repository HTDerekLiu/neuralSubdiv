clc; clear all; close all;
addpath('../utils/')

meshList = {'bunny_smooth'};
% avgVcList = [250];
for meshIdx = 1:length(meshList)
    meshName = meshList{meshIdx};
    numSubd = 2;
    numData = 10;
    variance = 100;
%     avgVc = avgVcList(meshIdx);
    avgVc = 300;

    % make folders
    folderList = {};
    for ii = 0:numSubd
        folder = strcat('./data_meshes/',meshName, "_", num2str(numData),'/subd',num2str(ii),'/');
        mkdir(folder)
        folderList{ii+1} = folder;
    end

    % load mesh
    [Vin,Fin] = load_mesh(strcat('../meshes/',meshName,'.obj'));
    Vin = normalizeUnitBox(Vin);

    % create subdivision meshes
    for ii = 1:numData
        nVc = avgVc + round(variance * (rand() - 0.5));
        [V,F,decInfo] = dec_EC(Vin,Fin,nVc,'QEM', 0.1,0.4,true);
        [VList,FList] = up_QEM_allLv(V,F,numSubd,Vin,Fin,decInfo);

        % save data
        [RV,IM,J] = remove_unreferenced(VList{1},FList{1});
        RF = IM(FList{1});
        subd(1).V = RV;
        subd(1).F = RF;

        % create eiejv
        [~,IM,~] = remove_unreferenced(VList{end},FList{end});
        for jj = 2:length(VList)
%             U = VList{jj-1};
%             W = FList{jj-1};

%             % get edge list
%             eiejv = [];
%             nV = size(U,1);
%             [Vnext,Fnext,S] = midPointUpsample(U,W,1);
%             [rIdx,cIdx,val] = find(S(nV+1:end,:)); 
%             assert(all(val == 0.5))
%             [rIdx, idx] = sort(rIdx+nV);
%             cIdx = cIdx(idx);
% 
%             rIdx = rIdx(1:2:end);
%             cIdx = reshape(cIdx,2,length(cIdx)/2)';
%             
%             fifjv = [];
%             for kk = 1:size(cIdx,1)
%                 vi = cIdx(kk,1);
%                 vj = cIdx(kk,2);
%                 [adjFi,~] = find(W == vi);
%                 [adjFj,~] = find(W == vj);
%                 adjF = intersect(adjFi, adjFj);
%                 assert(length(adjF) == 2)
%                 f1 = W(adjF(1),:);
%                 f2 = W(adjF(2),:);
%                 f1_vi = circshift(f1,-find(f1 == vi)+1); % circle shift f1 so "vi" is f1(1)
%                 f2_vi = circshift(f2,-find(f2 == vi)+1); % circle shift f2 so "vi" is f2(1)
%                 assert(f1_vi(1) == vi);
%                 assert(f2_vi(1) == vi);
%                 
%                 % check which one is f = [vi vj ?]
%                 if f1_vi(2) == vj 
%                     f_first = f1_vi;  % f_first = [vi, vj, ?]
%                     f_sec = f2_vi;    % f_sec   = [vi, ?, vj]
%                 elseif f2_vi(2) == vj
%                     f_first = f2_vi;  % f_first = [vi, vj, ?]
%                     f_sec = f1_vi;    % f_sec   = [vi, ?, vj]
%                 end
%                 assert(f_first(2) == vj)
%                 assert(f_sec(3) == vj)
%                 
%                 % assemble fifjv = [vi vj flap1 flap2 v]
%                 % the next order should be [vj vi flap2 flap1 v]
%                 fifj = [vi vj f_first(3) f_sec(2)];
%                 fifjv = [fifjv; fifj, rIdx(kk)];
%             end
%             eiejv = IM(eiejv);
%             fifjv = IM(fifjv); % RV(fifjv(k,1:4),:) is a flap position

%             % sanity check
%             [RV,IMtmp,~] = remove_unreferenced(VList{jj-1},FList{jj-1});
%             RF = IMtmp(FList{jj-1});
%             [RVnext,RFnext,S] = midPointUpsample(RV,RF,1);
%             dif = RV(eiejv(:,1),:)/2 + RV(eiejv(:,2),:)/2 - RVnext(eiejv(:,3),:);
%             assert(all(abs(dif(:)) < 1e-10))

            % save data
            [RV,IMtmp,J] = remove_unreferenced(VList{jj},FList{jj});
            RF = IMtmp(FList{jj});
            subd(jj).V = RV;
            subd(jj).F = RF;
%             subd(jj).fifjv = fifjv;

        end

        % output
        meshIdxStr = num2str(ii,'%03.f');

        writeOBJ(strcat(folderList{1},meshIdxStr,'.obj'), subd(1).V, subd(1).F);
        for jj = 2:length(VList)
            writeOBJ(strcat(folderList{jj},meshIdxStr,'.obj'), subd(jj).V, subd(jj).F);
            
%             fileID = fopen(strcat(folderList{jj},meshIdxStr,'_fifjv.txt'),'w');
%             fprintf(fileID,'%d %d %d %d %d\n',subd(jj).fifjv');
%             fclose(fileID);
        end
    end

end