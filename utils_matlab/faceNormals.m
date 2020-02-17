function FN = faceNormals(V, F)
% FACENORMALS computs per face unit normal 
% 
% FN = faceNormals(V, F)
%
% Inputs:
%   V |V| x 3 matrix of vertex positions
%   F |F| x 3 matrix of indices of triangle corners
% Outputs:
%   FN a |F| x 3 matrix of unit face normals

FN = cross(V(F(:,1),:)-V(F(:,2),:), V(F(:,1),:) - V(F(:,3),:));
FN = normalizerow(FN);