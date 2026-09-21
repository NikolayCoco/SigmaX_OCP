%DEMO_ANIMATION  Solve the qubit-reset problem while animating the iteration
%                (state / costate / control / cost panels) to a GIF or MP4.
%   Run from any folder:  >> run('demo_animation.m')
%
%   Animate    -> true   records every iteration to 'iteration.gif' (or .mp4)
%   showLive   -> true   also leaves a live figure on screen
%   format     -> 'gif'  (universally viewable) or 'mp4'
%
%   Use a modest number of iterations so the recording is fast.

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src'));

%% Model and solver
model = ocp.QubitReset('tau',1.5,'N',60,'gamma',10);
s = solver.GradientSolver(model);
s.direction  = 'gradient';
s.lineSearch = 'armijo';
s.maxIter    = 1e5;          % keep modest for a quick GIF
s.tolGrad    = 1e-4;

%% Visualizer / animator
v = vis.Visualizer('animate', true, 'format', 'gif', ...
                   'videoFile', 'iteration.gif', 'fps', 12, 'showLive', true);
tg = model.timeGrid(model.N);
v.init(model, tg);
s.observer = @(iter, J, X, Mu, U, tgrid) v.snapshot(iter, J, X, Mu, U, tgrid);

%% Solve (the observer records/animates every iteration)
fprintf('===== solving while animating =====\n');
res = s.solve();
v.finish();

fprintf('converged=%d (%s)  iter=%d  J=%.6e\n', ...
        res.converged, res.reason, res.iterations, res.J);
fprintf('animation written to: %s\n', v.videoFile);
