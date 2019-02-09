function [ X, Y, Z ] = problemOneSolver( rightHandSide )
% problemOneSolver solves the poisson equation on the defined domain
% leveraging the poisson equation on the given domain with step
% size of 1. Argument rightHandSide is a function pointer to a matlab
% function

% DE Operator Matrix
D = 4 * eye(7);
D(1,2) = -1;
D(1,5) = -1;
D(2,3) = -1;
D(2,6) = -1;
D(3,4) = -1;
D(5,6) = -1;
D(5,7) = -1;
D(2,1) = -1;
D(3,2) = -1;
D(4,3) = -1;
D(6,5) = -1;
D(7,5) = -1;

% X and Y Coordinates for evaluating right hand side
X = [1; 2; 3; 4; 1; 2; 1];
Y = [1; 1; 1; 1; 2; 2; 3];

% Actually evaluate the rhs at points
rhsVector = rightHandSide(X,Y);

% Solve for solution vector
solutionVector = D \ rhsVector;

% define the mesh grid for embedding solution and for visualization
[X,Y] = meshgrid(0:5, 0:4);

Z = [ 0, 0, 0, 0, 0, 0; 
    0, solutionVector(1), solutionVector(2), solutionVector(3), solutionVector(4), 0;
    0, solutionVector(5), solutionVector(6), 0, 0, 0;
    0, solutionVector(7), 0, 0, 0, 0;
    0, 0, 0, 0, 0, 0];

% Remove x and y coordinates that are not within the domain or boundary
zeroIndices = find(~conv2(Z, [1 1 1; 1 0 1; 1 1 1], 'same'));

X(zeroIndices) = NaN;
Y(zeroIndices) = NaN;
