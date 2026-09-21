%DEMO_PROLONG  Homotopy continuation (prolongation) on the qubit-reset problem.
%   Run from any folder:  >> run('demo_prolong.m')
%
%   The terminal cost |p(tau)| makes the gradient vanish w.r.t. the initial
%   control when the horizon is long.  Continuing from an EASY sub-problem to
%   the problem of interest (warm-starting each step) cures this.  Two
%   continuation families are exposed by the model:
%
%       continuName = 'tau'   : grow the horizon time  (default in this demo)
%       continuName = 'kappa' : grow the terminal-cost weight 0 -> 1
%
%   See descriptor/main.tex (section "regularization & continuation").

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src'));

%% 1. Model and continuation path
model = ocp.QubitReset('tau', 1.5, 'N', 5e2, 'gamma', 10, 'lambdaMax', [2; 1.5]);
model.homotopyName = 'tau';                     % continue on the horizon

% start from a short horizon, grow to the target tau (the prolongation path)
tauStart = 0.1;  tauEnd = model.tau;  steps = 14;
path = linspace(tauStart, tauEnd, steps);      % ascending

%% 2. Wrap the model and solve along the path
c = continuation.HomotopyContinuation(model, path);
c.solver.direction  = 'gradient';
c.solver.lineSearch = 'armijo';
c.solver.maxIter    = 1e3;
c.solver.tolGrad    = 1e-5;

fprintf('\n===== homotopy on %s, %d steps =====\n', model.homotopyName, numel(path));
res = c.run();

%% 3. Plot the cost and the final control
figure('Color', 'w', 'Position', [80 80 1100 420]);

subplot(1, 2, 1);
plot(path, [res.results.J], 'o-', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel([model.homotopyName ' (continuation parameter)']);
ylabel('cost'); grid on;
title('cost along the continuation path');

subplot(1, 2, 2);
tg = model.timeGrid(model.N);
plot(tg, res.U(1,:), 'r-', 'LineWidth', 1.5); hold on;
plot(tg, res.U(2,:), 'b-', 'LineWidth', 1.5);
xlabel('time'); ylabel('control');
legend('\lambda_x', '\lambda_z', 'Location', 'best'); grid on;
title('final control');

fprintf('final cost = %.6e (from %.6e)\n', res.J, res.results(1).J);
