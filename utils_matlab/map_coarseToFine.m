function [b, vIdx] = map_coarseToFine(b, vIdx, decInfo)
    assert(all(b <= 1))
    assert(all(b >= 0))
    assert(abs(sum(b) - 1) < 1e-10)
    for ii = 1:length(decInfo)
        dec = decInfo(end-ii+1);
        idx = find(vIdx == dec.vi);
        if length(idx) > 0 % if contained
            [b, vIdx] = map_coarseToFine_single(b, vIdx, dec);
        end
    end
end

function [b, vIdx] = map_coarseToFine_single(b,vIdx, dec)
    assert(all(b <= 1))
    assert(all(b >= 0))
    assert(abs(sum(b) - 1) < 1e-10)
    
    % check query face is in the UV domain (optional)
    v1 = find(dec.subsetVIdx == vIdx(1));
    v2 = find(dec.subsetVIdx == vIdx(2));
    v3 = find(dec.subsetVIdx == vIdx(3));
%     [adjv1,~] = find(dec.FUV == v1);
%     [adjv2,~] = find(dec.FUV == v2);
%     [adjv3,~] = find(dec.FUV == v3);
%     fIdx = intersect(intersect(adjv1, adjv2), adjv3);
%     assert(length(fIdx) == 1)

    % compute query UV
    queryUV = b(1) * dec.UV(v1,:) ...
        + b(2) * dec.UV(v2,:) ...
        + b(3) * dec.UV(v3,:);
    
    % compute barycentric coordinates for all triangle FUV_pre
    B = computeBarycentric(queryUV, dec.UV_pre, dec.FUV_pre);
    minD = -1e-7;
    idxToFUV_pre = find(B(:,1)>=minD & B(:,2)>=minD & B(:,3)>=minD & ...
        B(:,1)<=1-minD & B(:,2)<=1-minD & B(:,3)<=1-minD);
    
%     if length(idxToFUV_pre) == 0 % due to numerical error
%         minD = -1e-7;
%         idxToFUV_pre = find(B(:,1)>=minD & B(:,2)>=minD & B(:,3)>=minD);
%         assert(length(idxToFUV_pre) == 1);
%     end
    idxToFUV_pre = idxToFUV_pre(1);
    
    b = B(idxToFUV_pre,:);
    vIdx = dec.subsetVIdx(dec.FUV_pre(idxToFUV_pre,:));
    
    % make sure b is a valid barycentric
    negIdx = find(b < 0);
    if length(negIdx) > 0
        negVal = min(b);
        b = b - negVal;
    end
    b = b / sum(b);
end