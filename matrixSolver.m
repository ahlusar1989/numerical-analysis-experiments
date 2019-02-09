function [X, Y, Z] = matrixSolver(inputStepSize, domain, rhsEquation)
% Solve poisson equation with domain given by unit squares in
% matrix. The differential equation is then represented by a
% matrix system of equations to solve.

% Stride length
steps = round(1 / inputStepSize);

% Refining the domain using the kronecker product - 
% easily could have scaled this iteratively
refinedDomain = kron(domain, ones(steps));

% Fix x axis missing expanded values and boundaries
leftSide = kron(refinedDomain(:, 1), ones(1, steps-1));
leftBoundary = zeros(size(leftSide, 1), 1);
rightBoundary = leftBoundary;
refinedDomain = [leftBoundary, leftSide, refinedDomain, rightBoundary];

% Fix y axis missing expanded values and boundaries
belowboundary = kron(refinedDomain(1,:), ones(steps-1, 1));
topBoundary = zeros(1, size(belowboundary,2));
bottomBoundary = topBoundary;
refinedDomain = [bottomBoundary; belowboundary; refinedDomain; topBoundary];

% Find nonzero elements on the domain
[yPoints, xPoints] = find(refinedDomain);

% Scale x and y points based on step size
xPoints = inputStepSize * xPoints;
yPoints = inputStepSize * yPoints;

% Initialize operator matrix with diagonals
operatorMatrix = sparse(1:numel(xPoints), 1:numel(xPoints), ...
    4 * ones(numel(xPoints), 1));

% Iterate through each neighborhood and assign -1 where necessary
for i=1:numel(xPoints)
    currentPoint = [xPoints(i), yPoints(i)];
    
    % Set up coordinates of surrounding points
    above = [currentPoint(1), currentPoint(2) + inputStepSize];
    below = [currentPoint(1), currentPoint(2) - inputStepSize];
    left = [currentPoint(1) - inputStepSize, currentPoint(2)];
    right = [currentPoint(1) + inputStepSize, currentPoint(2)];
    
    % Vector of surrounding points for easy iteration
    surroundingPoints = [above; below; left; right];
    
    % Retrieve individual indices for each 
    for j=1:numel(surroundingPoints) / 2
        idx = findPoint(xPoints, yPoints, [surroundingPoints(j, 1); surroundingPoints(j,2)]);
        operatorMatrix(i, idx) = -1;
    end
end

% Create right hand side vector
rhsVector = inputStepSize.^2 * rhsEquation(xPoints, yPoints);

% Solve the system
solutionMatrix = operatorMatrix \ rhsVector;

% Coordinates plotted for visualization
[X, Y] = meshgrid(0:inputStepSize:size(domain, 2)+1, ...
    0:inputStepSize:size(domain, 1)+1);

Z = refinedDomain;
for i=1:numel(solutionMatrix)
    Z(find(Z == 1, 1)) = solutionMatrix(i);
end

% Remove values for x and y that are not internal or boundary points
zeroIndices = find(~conv2(Z, [1 1 1; 1 0 1; 1 1 1], 'same'));
X(zeroIndices) = NaN;
Y(zeroIndices) = NaN;
    

end