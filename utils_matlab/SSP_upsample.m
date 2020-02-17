function [VList,FList] = SSP_upsample(Vb,Fb,numSubd,Vin,Fin,decInfo)
    VList = {};
    FList = {};
    
    % update Vb
    [vList,bC,bF] = barycentricFromS(Fb,speye(size(Vb,1)));
    tmpV = zeros(size(bC,1), 3);
    parfor ii = 1:size(bC,1)
        if mod(ii, 100) == 0
            fprintf('QEM_up: %d/%d\n', [ii,size(bC,1)]);
        end
        [b, vIdx] = map_coarseToFine(bC(ii,:), bF(ii,:), decInfo);
        tmpV(ii,:) = b * Vin(vIdx,:);
    end
    Vb(vList,:) = tmpV;
    
    % save Vb
    VList{1} = Vb;
    FList{1} = Fb;

    % compute subdivided meshes
    visitedV = [];
    for sIdx = 1:numSubd
        [V,F,S] = midPointUpsample(Vb,Fb,sIdx);
        [vList,bC,bF] = barycentricFromS(Fb,S);
        
        if length(visitedV) > 0
            maxIdx = max(visitedV);
            bC(vList <= maxIdx,:) = [];
            bF(vList <= maxIdx,:) = [];
            vList(vList <= maxIdx) = [];
        end

        tmpV = zeros(size(bC,1), 3);
        parfor ii = 1:size(bC,1)
            if mod(ii, 100) == 0
                fprintf('QEM_up: %d/%d\n', [ii,size(bC,1)]);
            end
            [b, vIdx] = map_coarseToFine(bC(ii,:), bF(ii,:), decInfo);
            tmpV(ii,:) = b * Vin(vIdx,:);
        end
        V(vList,:) = tmpV;
        V(visitedV,:) = VList{sIdx}(visitedV,:);
        
        visitedV = [visitedV; vList];
        VList{sIdx+1} = V;
        FList{sIdx+1} = F;
    end
end

function [vList,bC,bF] = barycentricFromS(F,S)
    vList = unique(F);
    vList = [vList', 1+size(S,2):size(S,1)]';
    bC = zeros(length(vList), 3);
    bF = zeros(length(vList), 3, 'uint32');
    for ii = 1:length(vList)
        if mod(ii, 500) == 0
            fprintf('barycentricFromS: %d/%d\n', [ii,length(vList)]);
        end
        v = vList(ii);
        [~, adjV, b] = find(S(v,:));
        
        % find the barycentric face
        if length(adjV) == 1 
            [f,~] = find(F == adjV);
        elseif length(adjV) == 2
            [f1,~] = find(F == adjV(1));
            [f2,~] = find(F == adjV(2));
            f = intersect(f1, f2);
        elseif length(adjV) == 3
            [f1,~] = find(F == adjV(1));
            [f2,~] = find(F == adjV(2));
            [f3,~] = find(F == adjV(3));
            f = intersect(intersect(f1, f2), f3);
        end
        f = f(1); 
        
        % get barycentric entries
        [~,ia,ib] = intersect(adjV, F(f,:));
        [~, iaInv] = sort(ia);
        adjVInFinf = ib(iaInv);

        bF(ii,:) = F(f,:);
        bC(ii,adjVInFinf) = b;
    end
end