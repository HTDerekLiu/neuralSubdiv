function [UV,FUV,subsetVIdx,subsetFIdx,b,bc] = lscmOneRing(V,F,e)
% LSCMONRING use LSCM "Spectral Conformal Parameterization" [Mullen et al. 2008] but we only perform the minimization of the conformal energy (instead of eigen solve)
% to flatten the one ring patch of an edge e
% 
% Inputs:
%     V: nV x 3 vertex positions
%     F: nF x 3 face list
%     e: length 2 of an edge
%     
% Outputs:
%     UV: |VLIdx| x 2 UV positions of vertices VLIdx
%     FUV: |FL| x 3 face indices for the UV
%     subsetVIdx: |VLIdx| x 1 indices of the onering vertices (including the edge itself)
%     subsetFIdx: |FL| x 1 indices of the onering faces
%     b: length 2 of constrained vertex indices (in (UV, FUV))
%     bc: |b| x 2 of constrained vertex positions
%
% Notes:
%     - the faces of F(subsetFIdx,:) are the face in FUV, but since the
%     length of total vertices are different FUV ~= F(subsetFIdx,:)
%     - {UV, FUV} will be the UV triangle mesh of {V, F(subsetFIdx,:)}
    

vi = e(1);
vj = e(2);
[adjFi,~] = find(F == vi);
[adjFj,~] = find(F == vj);
adjF = unique([adjFi; adjFj]);

lij = norm(V(vi,:) - V(vj,:));

FL = F(adjF,:);
[VL,IM,subsetVIdx] = remove_unreferenced(V,FL);
FUV = IM(FL);
viL = find(subsetVIdx == vi);
vjL = find(subsetVIdx == vj);

b = [viL, vjL];
bc = [0,0; 5*lij, 0];
UV = lscm(VL,FUV,b,bc);
subsetFIdx = adjF;





