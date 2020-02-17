function adjF = vertexFaceAdjacency(F)
  % VERTEXFACEADJACENCYLIST constructs a list indicates the
  % indices of adjacenct faces
  %
  % adjF = vertexFaceAdjacencyList(F)
  %
  % Input:
  %   F   |F| x 3   list of face indices
  % Output:
  %   adjF |V| list where adjF{ii} outputs the adjacent face
  %   indices of a vertex

  i = (1:size(F,1))';
  j = F;
  VT = sparse([i i i],j,1);
  
  indices = 1:size(F,1);
  adjF = cell(size(VT,2),1);
  for ii = 1:size(VT,2)
    adjF{ii} = indices(logical(VT(:,ii)));
  end

end