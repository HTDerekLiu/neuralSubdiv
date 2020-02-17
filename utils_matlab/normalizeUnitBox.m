function V = normalizeUnitBox(V)
% NORMALIZEUNITBOX normalizes a given mesh to an unit bounding box
%
% V = normalizeUnitBox(V,F)
%
% Inputs:
%   V |V| x 3 matrix of vertex positions
% Outputs:
%   V a new matrix of vertex positions who is in a unit bounding box

V = V - min(V);
V = V / max(V(:));