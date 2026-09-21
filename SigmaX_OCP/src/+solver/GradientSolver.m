classdef GradientSolver < handle
%GRADIENTSOLVER  Direct-method gradient solver for an AbstractOCP.
%
%   The solver iterates on the (discretized) control and only talks to the
%   model through the AbstractOCP interface, so the *problem* is completely
%   replaceable.  Available algorithmic knobs:
%
%   direction      string      'gradient' (default), 'adagrad', 'rmsprop',
%                              'momentum' (classical momentum), 'nestorov',
%                              'adam'  -> history-accumulated direction methods
%   lineSearch     string      'simple', 'armijo', 'wolfe'
%   interpolation  string      'bisection'
%   useProjection  logical     project control onto its box each iteration
%   kktActiveSet   logical     zero the gradient at active bounds (projected
%                              gradient / KKT active-set handling of the path
%                              constraints)
%
%   Solve with
%       s = ocp.GradientSolver(model);
%       s.direction = 'adam';
%       res = s.solve();
%   and inspect res.U, res.J, res.hist.

    properties
        model            % AbstractOCP
        direction   = 'gradient'
        lineSearch  = 'armijo'
        interpolation = 'bisection'
        useProjection   = true
        kktActiveSet    = false
        maxIter         = 1000
        minIter         = 1
        tolGrad   = 1e-4
        tolRel    = 1e-6
        beta1 = 0.9
        beta2 = 0.999
        eps   = 1e-8
        alpha0   = 0.5
        alphaMin = 1e-8
        alphaMax = 1
        rho    = 0.1
        sigma  = 0.4
        N      = 100
        verbosity = 1
        observer   = []   % optional handle: @(iter,J,X,Mu,U,tg) called each iteration
    end

    properties (Access = private)
        m = [];        % first moment (velocity)
        v = [];        % second moment (accumulated squared gradient)
        k = 0          % iteration counter (for Adam bias correction)
        ls             % a LineSearch instance (stateless)
    end

    methods
        function obj = GradientSolver(model, varargin)
            if nargin >= 1, obj.model = model; end
            obj.ls = solver.LineSearch;
            obj.set(varargin{:});
        end

        function obj = set(obj, varargin)
            for i = 1:2:numel(varargin)
                obj.(varargin{i}) = varargin{i+1};
            end
        end

        function res = solve(obj, U0, tg)
        %SOLVE  Run the gradient iteration and return a result struct.
            model = obj.model;
            if nargin < 3
                tg = model.timeGrid(obj.N);
            end
            if nargin < 2 || isempty(U0)
                U0 = model.initialControl(tg);
            end
            n = numel(tg);
            d = model.nControl;

            obj.k = 0; obj.m = []; obj.v = [];   % reset accumulators

            if obj.useProjection, U = model.project(U0); else, U = U0; end

            hist.J = []; hist.gnorm = []; hist.alpha = [];
            conv = false; reason = '';
            bestJ = inf; bestU = U; bestX = []; bestMu = [];

            for iter = 1:obj.maxIter
                [J, X, Mu, G] = model.costGradient(tg, U);

                % ---- KKT / active-set (projected gradient) ----
                if obj.useProjection && obj.kktActiveSet
                    G = obj.applyActiveSet(U, G);
                end

                if J < bestJ
                    bestJ = J; bestU = U; bestX = X; bestMu = Mu; bestG = G;
                end

                if ~isempty(obj.observer)
                    obj.observer(iter, J, X, Mu, U, tg);
                end

                gnorm = sqrt(trapz(tg, sum(G.^2, 1)));
                hist.J(end+1)    = J;
                hist.gnorm(end+1)= gnorm;

                if iter >= obj.minIter && gnorm < obj.tolGrad
                    conv = true; reason = 'gradient small'; break;
                end

                % ---- direction and projected step ----
                ddir = obj.computeDirection(model, tg, U, G);
                dphi0 = trapz(tg, sum(G .* ddir, 1));

                if dphi0 < 0
                    alpha = obj.lineSearchStep(model, tg, U, ddir, G);
                else
                    alpha = 0;
                end

                U_old = U;
                if obj.useProjection
                    U = model.project(U + alpha*ddir);
                else
                    U = U + alpha*ddir;
                end
                hist.alpha(end+1) = alpha;

                % relative / step-size convergence
                delta = U - U_old;
                stepNorm = sqrt(trapz(tg, sum(delta.^2, 1)));
                if iter >= obj.minIter && stepNorm < obj.tolGrad
                    conv = true; reason = 'step small'; break;
                end
                if iter >= obj.minIter && iter > 1 ...
                        && abs(hist.J(end) - hist.J(end-1)) < obj.tolRel*abs(hist.J(end-1))
                    conv = true; reason = 'cost unchanged'; break;
                end

                if obj.verbosity > 0 && (mod(iter, 50) == 0 || iter == 1)
                    fprintf('[%s] iter %4d  J = %.6e  |grad| = %.3e  alpha = %.3e\n', ...
                            obj.direction, iter, J, gnorm, alpha);
                end
            end

            if iter == obj.maxIter, reason = 'max iteration'; end

            res.U = bestU; res.J = bestJ; res.X = bestX; res.Mu = bestMu;
            res.G = bestG; res.hist = hist; res.converged = conv;
            res.reason = reason; res.iterations = min(iter, obj.maxIter);
        end
    end

    methods (Access = private)
        function G = applyActiveSet(obj, U, G)
        %APPLYACTIVESET  Zero gradient components pushing outward at a bound.
            [lo, hi] = obj.model.controlBounds();
            atUp = U >= hi - 1e-12;
            atLo = U <= lo + 1e-12;
            outward = (atUp & G < -1e-12) | (atLo & G > 1e-12);
            G(outward) = 0;
        end

        function ddir = computeDirection(obj, model, tg, U, G)
        %COMPUTEDIRECTION  History-accumulated descent direction from grad J.
            method = lower(obj.direction);
            obj.k = obj.k + 1;
            switch method
                case {'grad','gradient'}
                    ddir = -G;
                case 'adagrad'
                    if isempty(obj.v), obj.v = zeros(size(G)); end
                    obj.v = obj.v + G.^2;
                    ddir = -G ./ (sqrt(obj.v) + obj.eps);
                case 'rmsprop'
                    if isempty(obj.v), obj.v = zeros(size(G)); end
                    obj.v = obj.beta2*obj.v + (1-obj.beta2)*G.^2;
                    ddir = -G ./ (sqrt(obj.v) + obj.eps);
                case {'momentum','cm'}
                    if isempty(obj.m), obj.m = zeros(size(G)); end
                    obj.m = obj.beta1*obj.m + (1-obj.beta1)*G;
                    ddir = -obj.m;
                case {'nestorov','nag'}
                    if isempty(obj.m), obj.m = zeros(size(G)); end
                    Ula = U + obj.beta1*obj.m;                 % look-ahead point
                    [~, ~, ~, Gla] = model.costGradient(tg, Ula);
                    obj.m = obj.beta1*obj.m + (1-obj.beta1)*Gla;
                    ddir = -obj.m;
                case 'adam'
                    if isempty(obj.m), obj.m = zeros(size(G)); end
                    if isempty(obj.v), obj.v = zeros(size(G)); end
                    obj.m = obj.beta1*obj.m + (1-obj.beta1)*G;
                    obj.v = obj.beta2*obj.v + (1-obj.beta2)*G.^2;
                    mhat = obj.m/(1 - obj.beta1^obj.k);
                    vhat = obj.v/(1 - obj.beta2^obj.k);
                    ddir = -mhat ./ (sqrt(vhat) + obj.eps);
                otherwise
                    error('GradientSolver: unknown direction "%s"', obj.direction);
            end
        end

        function alpha = lineSearchStep(obj, model, tg, U, ddir, G)
        %LINESEARCHSTEP  Wrap the LineSearch class with the solver options.
            s.rho      = obj.rho;
            s.sigma    = obj.sigma;
            s.alpha0   = obj.alpha0;
            s.alphaMin = obj.alphaMin;
            s.alphaMax = obj.alphaMax;
            alpha = obj.ls.search(obj.lineSearch, obj.interpolation, ...
                                  model, tg, U, ddir, G, s);
        end
    end
end
