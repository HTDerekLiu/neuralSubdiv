function  [V,F,decInfo] = SSP_decimation(V,F,tarV, ...
    method, triQ_threshold, dotn_threshold, randomEC)

    % TODO: check tetrahedron
    % TODO: extend to mesh with boundaries

    % parse parameters 
    if(~exist('method','var'))
        method = 'QEM'; % this could be 'QEM', 'midPoint'
    end
    if(~exist('triQ_threshold','var'))
        triQ_threshold = 0.2;
    end
    if(~exist('dotn_threshold','var'))
        dotn_threshold = 0.2;
    end
    if(~exist('randomEC','var'))
        randomEC = false;
    end

    % check whether input mesh is nonmanifold
    VCheck = is_vertex_nonmanifold(F);
    if any(VCheck)
        error('vertex non-manifold\n');
    end
    [~,ECheck] = nonmanifold_edges(F);
    if any(ECheck)
        error('edge non-manifold\n');
    end
    O = outline(F);
    if length(O) > 0
        error('mesh has boundary\n');
    end
    
    %% initialization
    E = edges(F);
    K = vertexCost(V,F,method);
    ECost = edgeCost(V,E,K,method);

    %% start edge collapse
    totalCollapse = size(V,1) - tarV;
    numCollapse = 0;
    stall = 0;
    maxStall = 500;
    while true
        if mod(numCollapse, 100) == 0
            fprintf('collapse progress %d / %d\n', [numCollapse totalCollapse])
        end
        stall = stall + 1;
        
        % get edge with the min cost
        if randomEC
            % random edge collapses
            idxList = randperm(length(ECost));
            idxList = idxList(1:100);
            ECost_subset = ECost(idxList);
            [~,e] = min(ECost_subset);
            e = idxList(e);
        else
            [~,e] = min(ECost);
        end

        % CHECK: if the edge is degenerated
        if E(e,1) == E(e,2) 
            E(e,:) = [];
            ECost(e) = [];
            stall = 0;
            continue;
        end

        % save previous mesh
        Vpre = V;
        Fpre = F;

        % compute parameterization
        [UV_pre,FUV_pre,subsetVIdx,subsetFIdx,b,bc] = lscmOneRing(V,F,E(e,:));
        UV = UV_pre;
        FUV = FUV_pre;
        assert(subsetVIdx(b(1)) == E(e,1)) % make sure b(1) corresponds to vi

        % CHECK: check lscm self folding (before collapse)
        angAll = internalangles(UV_pre,FUV_pre);
        idx1 = find(FUV_pre == b(1));
        ang1 = sum(angAll(idx1));
        idx2 = find(FUV_pre == b(2));
        ang2 = sum(angAll(idx2));
        if (ang1 - 2*pi > 1e-10) || (ang2 - 2*pi > 1e-10)
            fprintf('lscm self folding \n');
            ECost(e) = inf;
            if stall > maxStall break; end
            continue;
        end
        
        % get adjacent faces of this edge
        vi = E(e,1);
        vj = E(e,2);
        [adjFi,~] = find(F == vi);
        [adjFj,~] = find(F == vj);
        adjF = unique([adjFi; adjFj]);
        
        % CHECK: check link condition
        Nvi = unique(F(adjFi,:));
        Nvj = unique(F(adjFj,:));
        if length(intersect(Nvi, Nvj)) ~= 4 % 4 includes vi, vj
            fprintf('link condition fail \n');
            ECost(e) = inf;
            if stall > maxStall break; end
            continue;
        end

        % compute face normals before edge collapse
        FN_prev = cross(V(F(adjF,1),:)-V(F(adjF,2),:), V(F(adjF,1),:) - V(F(adjF,3),:));
        FN_prev = normalizerow(FN_prev);
      
        % reconnect faces
        F(F == vj) = vi;
        tmp = [F, F(:,1)];
        [delF, ~] = find(diff(tmp,1,2) == 0); % find faces that to be deleted
        assert(length(delF) == 2)

        % reconnect faces in the UV domain
        FUV(FUV == b(2)) = b(1);
        tmp = [FUV, FUV(:,1)];
        [delFUV, ~] = find(diff(tmp,1,2) == 0); % find faces that to be deleted
        assert(length(delFUV) > 0)

        % move vertex i
        viPos = V(vi,:);
        vjPos = V(vj,:);
        [optvi, ~] = optPos(K(:,:,vi) + K(:,:,vj),V(vi,:),V(vj,:), method); 
        V(vi,:) = optvi;

        % move vertex i in the UV domain
        subsetFIdx(subsetFIdx == delF(1)) = [];
        subsetFIdx(subsetFIdx == delF(2)) = [];
        lb = subsetVIdx;
        lb(b) = [];
        lbc = UV_pre;
        lbc(b,:) = [];
        [lV,lIM,lJ] = remove_unreferenced(V,F(subsetFIdx,:));
        lF = lIM(F(subsetFIdx,:));
        lb = lIM(lb);
        [U,~] = lscm(lV,lF,lb',lbc);
        newUV = U;
        newUV(lb,:) = [];
        assert(length(newUV) == 2)
        UV(b(1),:) = newUV;

        % compute new normal
        FN_prev(adjF == delF(1),:) = [];
        adjF(adjF == delF(1)) = [];
        FN_prev(adjF == delF(2),:) = [];
        adjF(adjF == delF(2)) = [];
        FN_new = cross(V(F(adjF,1),:)-V(F(adjF,2),:), V(F(adjF,1),:) - V(F(adjF,3),:));
        FN_new = normalizerow(FN_new);
        
        % CHECK: triangle quality
        l0 = sqrt(sum((V(F(adjF,1),:)-V(F(adjF,2),:)).^2,2));
        l1 = sqrt(sum((V(F(adjF,2),:)-V(F(adjF,3),:)).^2,2));
        l2 = sqrt(sum((V(F(adjF,3),:)-V(F(adjF,1),:)).^2,2));
        x = (l0+l1+l2) ./ 2;
        delta = sqrt(x .* (x-l0) .* (x-l1) .* (x-l2));
        triQ = 4 * sqrt(3) * delta ./ (l0.^2 + l1.^2 + l2.^2);
        if any(triQ < triQ_threshold)
            fprintf('bad triangle quality\n');
            V = Vpre;
            F = Fpre;
            ECost(e) = inf;
            if stall > maxStall break; end
            continue;
        end

        % CHECK: adj face normals flip
        dotProd = sum(FN_new .* FN_prev,2);
        if any(dotProd < dotn_threshold) 
            fprintf('normal flipped\n');
            V = Vpre;
            F = Fpre;
            ECost(e) = inf;
            if stall > maxStall break; end
            continue;
        end

        % CHECK: UN face normals flip
        UV12 = UV(FUV(:,1),:)-UV(FUV(:,2),:);
        UV13 = UV(FUV(:,1),:)-UV(FUV(:,3),:);
        FNUV_new = UV12(:,1) .* UV13(:,2) - UV12(:,2) .* UV13(:,1);
        FNUV_new(delFUV) = [];
        UV12 = UV_pre(FUV_pre(:,1),:)-UV_pre(FUV_pre(:,2),:);
        UV13 = UV_pre(FUV_pre(:,1),:)-UV_pre(FUV_pre(:,3),:);
        FNUV_pre = UV12(:,1) .* UV13(:,2) - UV12(:,2) .* UV13(:,1);
        if any(FNUV_new < 0) || any(FNUV_pre < 0) % cannot flip due to flipped normal
            fprintf('UV triangle flipped\n');
            V = Vpre;
            F = Fpre;
            ECost(e) = inf;
            if stall > maxStall break; end
            continue;
        end
        
        % CHECK: check lscm self folding (after collapse)
        tmpF = FUV;
        tmpF(delFUV,:) = [];
        angAll = internalangles(UV,tmpF);
        idx1 = find(tmpF == b(1));
        ang1 = sum(angAll(idx1));
        if (ang1 - 2*pi > 1e-10)
            fprintf('lscm self folding \n');
            V = Vpre;
            F = Fpre;
            ECost(e) = inf;
            if stall > maxStall break; end
            continue;
        end
        
        % CHECK: UV triangle quality
        l0 = sqrt(sum((UV(FUV(:,1),:)-UV(FUV(:,2),:)).^2,2));
        l1 = sqrt(sum((UV(FUV(:,2),:)-UV(FUV(:,3),:)).^2,2));
        l2 = sqrt(sum((UV(FUV(:,3),:)-UV(FUV(:,1),:)).^2,2));
        x = (l0+l1+l2) ./ 2;
        delta = sqrt(x .* (x-l0) .* (x-l1) .* (x-l2));
        triQ = 4 * sqrt(3) * delta ./ (l0.^2 + l1.^2 + l2.^2);
        triQ(delFUV) = [];
        if any(triQ < triQ_threshold)
            fprintf('bad UV triangle quality\n');
            V = Vpre;
            F = Fpre;
            ECost(e) = inf;
            if stall > maxStall break; end
            continue;
        end

        %% start post-collapsing
        % delete face
        F(delF,:) = [];
            
        % delete face in the UV space
        FUV(delFUV,:) = [];

        % reconnect edges
        E(E == vj) = vi;

        % update edge cost
        E(e,:) = [];
        ECost(e) = [];

        % update vertex quadric
        [ECost, K] = updateEdgeCost(V,E,vi,vj,K,ECost, method);

        % save decimation information for upsampling
        numCollapse = numCollapse + 1;
        decInfo(numCollapse).vi = vi;
        decInfo(numCollapse).b = b; % note that b(1) cprresponds to vi
        decInfo(numCollapse).UV = UV;
        decInfo(numCollapse).FUV = FUV;
        decInfo(numCollapse).UV_pre = UV_pre;
        decInfo(numCollapse).FUV_pre = FUV_pre;
        decInfo(numCollapse).subsetVIdx = subsetVIdx;
        stall = 0;

        %% check terminate
        if numCollapse == totalCollapse
            break;
        end
        if stall > 500
            break;
        end
    end
end

%% utility function
function KV = vertexCost(V,F, method)
    switch method 
        case {'QEM', 'WQEM'}
            FN = faceNormals(V,F);
            FA = doublearea(V,F) / 2;
            KF = zeros(4,4,size(F,1));
            for ii = 1:size(F,1)
                d = - FN(ii,:) * V(F(ii,1),:)';
                p = [FN(ii,:), d]';
                KF(:,:,ii) = FA(ii) * (p * p');
            end
            adjFList = vertexFaceAdjacencyList(F);
            KV = zeros(4,4,size(V,1));
            for ii = 1:size(V,1)
                adjF = adjFList{ii};
                KV(:,:,ii) = sum(KF(:,:,adjF),3);
            end
        case 'midPoint'
            KV = zeros(4,4,size(V,1));
        otherwise
            error('bad method')
    end
    
end


%%
function ECost = edgeCost(V,E,K_v,method)
    switch method
        case {'QEM'}
            ECost = zeros(size(E,1),1);
            for e = 1:size(E,1)
                % K_e = K_v1 + K_v2
                v1Idx = E(e,1); 
                v2Idx = E(e,2);
                Ke = K_v(:,:,v1Idx) + K_v(:,:,v2Idx); 

                % v'Kv = v'Av + 2b'v + c 
                A = Ke(1:3, 1:3);
                b = Ke(1:3, 4);
                c = Ke(4, 4);

                % optimal point A*p = -b
                p = A \ -b;
                ECost(e) = [p;1]' * Ke * [p;1];

                if isnan(ECost(e)) || isinf(ECost(e))
                    ECost(e) = inf; % set inf or nan to inf
                end
            end
        case 'midPoint'
            ECost = sqrt(sum((V(E(:,1),:) - V(E(:,2),:)).^2,2));
    end
end
    
%%
function [p, cost] = optPos(Ke, vi, vj, method)
    switch method 
        case {'QEM'}
            % v'Kv = v'Av + 2b'v + c 
            A = Ke(1:3, 1:3);
            b = Ke(1:3, 4);
            c = Ke(4, 4);
            vK = A \ -b;
            vK = vK';

            costK = [vK,1] * Ke * [vK,1]';
            costi = [vi,1] * Ke * [vi,1]';
            costj = [vj,1] * Ke * [vj,1]';
            costMid = [(vi+vj)./2,1]  * Ke * [(vi+vj)./2,1]';

            [cost,idx] = min([costK, costi, costj, costMid]);
            posList = [vK; vi; vj; (vi+vj)./2];
            p = posList(idx,:);
        case 'midPoint'
            p = (vi+vj)/2;
            cost = nan;
    end
end

%%
function [ECost, K] = updateEdgeCost(V,E,vi,vj,K,ECost, method)

        switch method
            case {'QEM'}
                K(:,:,vi) = K(:,:,vi) + K(:,:,vj);
                [adjE, ~] = find(E == vi);
                for ii = 1:length(adjE)
                    eIdx = adjE(ii);
                    Ke = K(:,:,E(eIdx,1)) + K(:,:,E(eIdx,2)); 
                    [~, cost] = optPos(Ke, V(E(eIdx,1),:), V(E(eIdx,2),:), method);
                    ECost(eIdx) = cost;
                    if isnan(ECost(eIdx)) || isinf(ECost(eIdx))
                        ECost(eIdx) = inf; % set inf or nan to inf
                    end
                end
                K(:,:,vj) = nan;
                
            case 'midPoint'
                [adjE, ~] = find(E == vi);
                ECost(adjE) = sqrt(sum((V(E(adjE,1),:) - V(E(adjE,2),:)).^2,2));
        end
end