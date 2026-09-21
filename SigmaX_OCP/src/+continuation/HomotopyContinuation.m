classdef HomotopyContinuation < handle
%HOMOTOPYCONTINUATION  Abstract continuation (prolongation) wrapper.
%
%   Prolongation / homotopy continuation solves a difficult optimal control
%   problem by embedding it in a one-parameter family of problems and
%   marching the parameter from an easy problem to the problem of interest,
%   warm-starting each solve with the previous solution:
%
%       p_0 (easy)  ->  p_1  ->  ... ->  p_end = original problem
%
%   The generic continuation parameter is `model.homotopyValue`.  For the
%   qubit-reset problem two physical continuation families are supported:
%
%       * continue on the horizon tau   (homotopyName = 'tau')   -- shrinks
%         the time window so the terminal-cost gradient does not vanish at
%         the start (the "initial-position vanishing gradient" problem).
%       * continue on the terminal-cost weight kappa (0 -> 1) which adds a
%         regularizing running cost (homotopyName = 'kappa')  -- see the
%         descriptor's "regularization / continuation" section.
%
%   The wrapper is fully problem-agnostic: it only calls setHomotopy and the
%   underlying GradientSolver, so it can be reused for any AbstractOCP that
%   supports continuation.
%
%       c = solver.HomotopyContinuation(model, linspace(-1,1,41));  % e.g. tau path
%       r = c.run();
%       r.results(end).U, r.results(end).J

    properties
        model
        path            % vector of continuation values (ascending order)
        solver          % GradientSolver used for each inner solve
        N               % number of control samples (default from model)
        verbose = true
    end

    methods
        function obj = HomotopyContinuation(model, path, varargin)
            obj.model = model;
            obj.path  = path;
            if isempty(varargin)
                if isprop(model, 'N'), obj.N = model.N; else, obj.N = 100; end
                obj.solver = solver.GradientSolver(model);
            else
                if isa(varargin{1}, 'solver.GradientSolver')
                    obj.solver = varargin{1};
                else
                    obj.solver = solver.GradientSolver(model, varargin{:});
                end
            end
            if obj.model.homotopyValue ~= path(1)
                obj.model.setHomotopy(path(1));
            end
        end

        function res = run(obj)
        %RUN  March the continuation parameter and solve at each step.
            nP = numel(obj.path);
            tgCache = cell(1, nP);
            U = [];
            results = struct('U', {}, 'J', {}, 'params', {}, 'converged', {});
            for i = 1:nP
                obj.model.setHomotopy(obj.path(i));
                tg = obj.model.timeGrid(obj.N);        % pick up changed horizon
                tgCache{i} = tg;
                if isempty(U)
                    U = obj.model.initialControl(tg);
                end
                if obj.verbose
                    fprintf('== prolong step %d/%d : %s = %.4g ==\n', ...
                            i, nP, obj.model.homotopyName, obj.path(i));
                end
                sol = obj.solver.solve(U, tg);
                results(i) = struct('U', sol.U, 'J', sol.J, ...
                                    'params', obj.path(i), 'converged', sol.converged);
                U = sol.U;                              % warm start
            end
            res.results = results;
            res.path    = obj.path;
            res.tgrids  = tgCache;
            res.U       = results(end).U;
            res.J       = results(end).J;
        end
    end
end
