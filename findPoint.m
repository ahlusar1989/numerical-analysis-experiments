function [ index ] = findPoint( x, y, p )
%FINDPOINT Finds a index of point p in vectors x and y

% Get index of point in x
xIndices = find(x == p(1));

% Get index of point in y
index = xIndices(find(y(xIndices) == p(2)));

end

