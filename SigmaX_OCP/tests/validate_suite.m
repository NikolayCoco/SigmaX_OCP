function validate_suite(srcPath)
%VALIDATE_SUITE  Self-check for the gradient OCP toolbox.
%   ASCII-only so it runs under any MATLAB locale. Call:
%       validate_suite('absolute\path\to\src')
%
%   Checks: (1) the analytic gradient is a descent direction,
%           (2) the gradient solver decreases the cost (qubit + example),
%           (3) history-accumulated direction (Adam) also reduces the cost,
%           (4) the homotopy wrapper runs on the qubit model.

    addpath(srcPath);                 % parent dir of the package folders
    cc = onCleanup(@() rmpath(srcPath));
    fprintf('=== gradient OCP self check ===\n');

    %% 1. Analytic gradient is a descent direction (sign test)
    m = ocp.QubitReset('tau',0.15,'N',60,'gamma',10);
    tg = m.timeGrid(m.N);
    U  = repmat(0.6*m.lambdaMax, 1, m.N);      % strictly positive control
    ok1 = descentCheck(m, tg, U);
    m2 = ocp.ControlExample('N',60);  tg2 = m2.timeGrid(m2.N);
    ok1b = descentCheck(m2, tg2, 0.3*ones(1, m2.N));
    fprintf('[1] gradient is descent dir    : qubit=%s example=%s\n', ...
            yesno(ok1), yesno(ok1b));

    %% 2. Solver decreases the cost (few iterations)
    ok2a = solverDecreases(ocp.QubitReset('tau',0.15,'N',50), 60, 'qubit');
    ok2b = solverDecreases(ocp.ControlExample('N',50), 60, 'example');
    fprintf('[2] solver decreases cost      : qubit=%s example=%s\n', ...
            yesno(ok2a), yesno(ok2b));

    %% 3. Adam direction also reduces the cost
    ok3 = solverDecreasesDir(ocp.QubitReset('tau',0.15,'N',50), 60, 'adam', 'qubit');
    fprintf('[3] adam direction reduces cost: %s\n', yesno(ok3));

    %% 4. Homotopy continuation runs
    ok4 = homotopyRuns();
    fprintf('[4] homotopy continuation runs : %s\n', yesno(ok4));

    fprintf('=== done (all should be 1) ===\n');
end

function ok = descentCheck(m, tg, U)
    [J0, ~, ~, G] = m.costGradient(tg, U);
    U2 = m.project(U - 1e-3*G);                % one small descent step
    J2 = m.costGradient(tg, U2);
    ok = isfinite(J2) && J2 < J0;
end

function ok = solverDecreases(m, iters, tag)
    ok = solverDecreasesDir(m, iters, 'gradient', tag);
end

function ok = solverDecreasesDir(m, iters, dir, tag)
    if nargin < 4, tag = dir; end
    s = solver.GradientSolver(m);
    s.direction = dir; s.maxIter = iters; s.verbosity = 0; s.lineSearch = 'armijo';
    res = s.solve();
    if numel(res.hist.J) < 2, ok = false; return; end
    J0 = res.hist.J(1); Jf = res.hist.J(end);
    fprintf('   [%s] %s: J0=%.4e -> Jf=%.4e (iter %d)\n', ...
            tag, dir, J0, Jf, res.iterations);
    ok = isfinite(Jf) && Jf < J0;
end

function ok = homotopyRuns()
    m = ocp.QubitReset('tau',0.15,'N',40,'homotopyName','kappa');
    path = linspace(0.2, 1, 4);
    c = continuation.HomotopyContinuation(m, path);
    c.solver.maxIter = 25; c.solver.verbosity = 0;
    r = c.run();
    fprintf('   [homotopy] final J=%.4e over %d steps\n', r.J, numel(path));
    ok = isfinite(r.J);
end

function s = yesno(v)
    if v, s = 'PASS'; else, s = 'FAIL'; end
end
