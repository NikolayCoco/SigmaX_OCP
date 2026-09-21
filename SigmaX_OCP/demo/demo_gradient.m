%DEMO_GRADIENT  Solve the time-optimal qubit-reset problem with the gradient
%               toolbox, and plot the result.
%   Run from any folder:  >> run('demo_gradient.m')
%   (Comments are kept ASCII so the file runs under any MATLAB locale.)
%
%   The problem, the solver and its algorithms are described in
%   'descriptor/main.tex' (compile with xelatex).

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src'));

%% 1. Build the model (the OCP to be solved)
model = ocp.QubitReset('tau', 0.15, 'N', 100, ...
                       'gamma', 10, 'omega_c', 1, 'lambdaMax', [3; 2]);

%% 2. Choose a solver configuration
%   direction : 'gradient' | 'adagrad' | 'rmsprop' | 'momentum' | 'nestorov' | 'adam'
%   lineSearch: 'simple'    | 'armijo'  | 'wolfe'
s = solver.GradientSolver(model);
s.direction  = 'adam';
s.lineSearch = 'armijo';
s.maxIter    = 2000;
s.tolGrad    = 1e-5;
s.verbosity  = 1;

%% 3. Solve
fprintf('\n===== solving %s with %s =====\n', model.name, s.direction);
t0 = tic;
res = s.solve();
fprintf('elapsed %.2f s  |  converged=%d (%s)  |  iter=%d\n', ...
        toc(t0), res.converged, res.reason, res.iterations);

%% 4. Plot state, control and cost history
tg = model.timeGrid(model.N);
figure('Color', 'w', 'Position', [80 80 1200 420]);

subplot(1, 3, 1);
plot(tg, res.X(1,:), 'r-', 'LineWidth', 1.5); hold on;
plot(tg, res.X(2,:), 'b-', 'LineWidth', 1.5);
plot(tg, res.X(3,:), 'm-', 'LineWidth', 1.5);
xlabel('time'); ylabel('population');
legend('p_e', 'p_r', 'p_i', 'Location', 'best'); grid on;
title('state');

subplot(1, 3, 2);
plot(tg, res.U(1,:), 'r-', 'LineWidth', 1.5); hold on;
plot(tg, res.U(2,:), 'b-', 'LineWidth', 1.5);
xlabel('time'); ylabel('control');
legend('\lambda_x', '\lambda_z', 'Location', 'best'); grid on;
title(['control (direction: ' s.direction ')']);

subplot(1, 3, 3);
semilogy(res.hist.J, 'k-', 'LineWidth', 1.5);
xlabel('iteration'); ylabel('cost');
grid on; title('cost history');

fprintf('\ninitial cost = %.6e, final cost = %.6e\n', ...
        res.hist.J(1), res.hist.J(end));
