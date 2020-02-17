function [V,F,S] = midPointUpsample(Vin,Fin,numIter)
% midPointUpsample perform mid point upsampling
%
% [VV,FF,S] = midPointUpsample(V,F,1)
%
% Inputs:
%   V  #V by 3 list of vertex positions
%   F  #F by 3 list of face indices
%   numIter number of mid-point upsampling iterations
%
% Outpus:
%   VV #VV by 3 new vertex positions
%   FF #FF by 3 list of face indices
%   S  #VV by #V matrix computing VV = SS *V 

if nargin < 3
    numIter = 1;
end

V = Vin;
F = Fin;
S = speye(size(V,1));

for iter = 1:numIter
    
    % obtain new face connectivity
    hE = [F(:,[1,2]);F(:,[2,3]);F(:,[3,1])];
    [E,~,hE2E] = unique(sort(hE,2),'rows');
    
    nV = size(V,1);
    nF = size(F,1);
    nE = size(E,1);
    
    i3 = nV      + [1:nF]';
    i1 = nV+nF   + [1:nF]';
    i2 = nV+2*nF + [1:nF]';
    
    hEF = [ F(:,1) i3 i2 ; F(:,2) i1 i3 ; F(:,3) i2 i1 ; i1 i2 i3];
    hE2E = [(1:nV)'; hE2E+nV];
    F = hE2E(hEF);
    
    % obtain new vertex positions
    r = [1:nE, 1:nE]' + nV;
    c = E(:);
    v = ones(length(c),1) * 0.5;
    
    r = [r; [1:nV]'];
    c = [c; [1:nV]'];
    v = [v; ones(nV,1)];
        
    Snext = sparse(r,c,v,nE+nV,nV);
    S = Snext * S;
    
    V = S*Vin;

end
