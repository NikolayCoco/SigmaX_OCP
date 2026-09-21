%DEMO_EXAMPLE1  Solve a DIFFERENT optimal control problem with the SAME
%               gradient solver -- demonstrating that the OCP is replaceable.
%   Run from any folder:  >> run('demo_example1.m')
%
%   Problem (descriptor example 1):
%       min  J = int_0^1 0.5 ( x^2 + u^2 ) dt
%       s.t. x(0) = 10,  x' = -x^2 + u,  free terminal state
%
%   The model is ocp.ControlExample.  No solver code is changed.

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src'));

model = ocp.ControlExample('N', 200);
tg    = model.timeGrid(model.N);

% the same solver as for the qubit problem
s = solver.GradientSolver(model);
s.direction  = 'gradient';
s.lineSearch = 'armijo';
s.maxIter    = 500;
s.tolGrad    = 1e-6;

fprintf('===== solving %s =====\n', model.name);
res = s.solve();
fprintf('iter=%d  J=%.6e  converged=%d (%s)\n', res.iterations, res.J, ...
        res.converged, res.reason);

% analytical reference: costate lambda(1)=0, optimum u = lambda, x' = -x^2+u
% (the descriptor works out the first iterate u^(1) = 0.5*((10t+1)/11)^2 - 1)
figure('Color','w','Position',[100 100 900 380]);
subplot(1,2,1); plot(tg, res.U, 'b-', 'LineWidth', 1.5); grid on;
xlabel('time'); ylabel('control u'); title('control');
subplot(1,2,2); plot(tg, res.X, 'r-', 'LineWidth', 1.5); grid on;
xlabel('time'); ylabel('state x'); title('state');
