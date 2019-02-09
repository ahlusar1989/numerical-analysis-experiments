function [ X, Y, Z ] = problemOneSolverPartTwo( rightHandSide )
% problemOneSolver solves the poisson equation on the defined domain
% leveraging the poisson equation on the given domain with step
% size of 0.5. Argument rightHandSide is a function pointer to a matlab
% function


% DE Operator Matrix initialized here with specific indices
D = 4 * eye(43);
% Add -1 to neighbors to specific indices during iterative loop (below)
i = [1;1;2;2;2;3;3;3;4;4;4;5;5;5;6;6;6;7;7;8;8;8;9;9;9;9;10;10;10;10;
    11;11;11;11;12;12;12;12;13;13;13;13;14;14;14;15;15;15;16;16;16;16;
    17;17;17;17;18;18;18;18;19;19;19;19;20;20;20;21;21;22;22;22;23;23;
    23;23;24;24;24;24;25;25;25;25;26;26;26;27;27;27;28;28;28;28;29;29;
    29;29;30;30;30;31;31;32;32;32;33;33;33;33;34;34;34;35;35;35;36;36;
    36;36;37;37;37;38;38;38;39;39;39;39;40;40;40;41;41;42;42;42;43;43];

j = [2;8;1;9;3;2;10;4;3;11;5;4;12;6;5;13;7;6;14;1;9;15;2;10;16;8;3;11;
    17;9;4;12;18;10;5;13;19;11;6;14;20;12;7;21;13;8;16;22;9;17;23;15;10;
    18;24;16;11;19;25;17;12;20;26;18;19;13;21;20;14;15;23;27;16;24;28;22;
    17;25;29;23;18;26;30;24;19;31;25;22;28;32;23;29;33;27;24;30;34;28;
    25;31;29;26;30;27;33;35;28;34;36;32;29;37;33;32;36;38;33;37;39;35;
    34;40;36;35;39;41;36;40;42;38;37;43;39;38;42;41;39;43;42;40];

%construct DE matrix
for k=1:43
    D(i(k),j(k)) = -1;
end

domain = [1 1 1 1 1 1 1 1 1;
    1 1 1 1 1 1 1 1 1;
    1 1 1 1 1 1 1 1 1;
    1 1 1 1 1 0 0 0 0;
    1 1 1 1 1 0 0 0 0;
    1 1 1 0 0 0 0 0 0;
    1 1 1 0 0 0 0 0 0];

% Retrieve x and y points that will be evaluated
[yPoints, xPoints] = find(domain);
xPoints = 0.5 * xPoints;
yPoints = 0.5 * yPoints;


% Evaluate right hand side at points
rhsVector = 0.5^2 * rightHandSide(xPoints, yPoints);

% Solve the solution vector
solutionVector = D \ rhsVector;

% X and Y coordinates
[X,Y] = meshgrid(0:0.5:5, 0:0.5:4);

Z = [zeros(7,1), domain, zeros(7,1)];
Z = [zeros(1, 11); Z; zeros(1, 11)];
for i=1:numel(solutionVector)
    Z(find(Z == 1, 1)) = solutionVector(i);
end

% Remove x and y coordinates that are not within the domain or boundary
zeroIndices = find(~conv2(Z, [1 1 1; 1 0 1; 1 1 1], 'same'));

X(zeroIndices) = NaN;
Y(zeroIndices) = NaN;