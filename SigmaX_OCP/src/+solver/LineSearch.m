classdef LineSearch
%LINESEARCH  Non-exact (inexact) step-size rules for gradient-descent type
%            algorithms applied to an AbstractOCP.
%
%   Given a descent direction d (so that phi'(0) = <grad J, d> < 0), the
%   line search returns a step alpha such that the cost decreases
%   sufficiently.  Supported rules:
%
%       'simple'   pure backtracking (sufficient-decrease condition only)
%       'armijo'   Armijo-Goldstein: sufficient decrease + lower curvature bound
%       'wolfe'    Wolfe-Powell:  sufficient decrease + dphi >= sigma*dphi0
%
%   Interpolation when a step is rejected: 'bisection' (halve the step).
%
%   The search needs only the AbstractOCP interface (costGradient), so a
%   single LineSearch object serves any problem.
%
%   Example:
%       ls = solver.LineSearch;
%       alpha = ls.search('armijo','bisection', model, tg, U, d, G, opts);

    methods
        function alpha = search(obj, rule, interp, model, tg, U, d, G0, opts)
        %SEARCH  Return a step length alpha for the direction d.
        %   G0 is grad J at U; opts has .rho, .sigma, .alpha0, .alphaMin, .alphaMax.
            c1 = opts.rho;            % sufficient decrease parameter
            c2 = opts.sigma;          % curvature parameter (Wolfe)
            phi = @(a) obj.cost(model, tg, U, d, a, false);
            dph = @(a) obj.cost(model, tg, U, d, a, true);

            phi0  = phi(0);
            dphi0 = dph(0);
            if dphi0 >= 0
                alpha = 0;            % not a descent direction
                return;
            end

            alpha = opts.alpha0;
            iter  = 0; maxIter = 100;
            % ---- Phase 1: sufficient decrease (upper bracket) ----
            while phi(alpha) > phi0 + c1*alpha*dphi0
                alpha = obj.reduce(interp, alpha, opts.alphaMin);
                if alpha <= opts.alphaMin || iter > maxIter
                    alpha = opts.alphaMin;
                    break;
                end
                iter = iter + 1;
            end

            % ---- Phase 2 (Wolfe only): not too small (curvature) ----
            if strcmpi(rule, 'wolfe') && dphi(alpha) < c2*dphi0
                iter = 0;
                while dphi(alpha) < c2*dphi0
                    alpha = min(2*alpha, opts.alphaMax);
                    if alpha >= opts.alphaMax || iter > maxIter, break; end
                    iter = iter + 1;
                end
            end
        end

        function a = reduce(~, interp, alpha, alphaMin)
            %REDUCE  Safeguard a rejected step back toward 0.
            switch lower(interp)
                case {'bi','bisection','quad','quadratic'}
                    a = 0.5*alpha;    % bisection / simple quadratic shrinking
                otherwise
                    a = 0.5*alpha;
            end
            if a < alphaMin, a = alphaMin; end
        end

        function val = cost(~, model, tg, U, d, alpha, wantDeriv)
            %COST  Evaluate phi(alpha) or phi'(alpha) along the direction d.
            Ut = U + alpha*d;
            [J, ~, ~, G] = model.costGradient(tg, Ut);
            if wantDeriv
                val = trapz(tg, sum(G .* d, 1));   % <grad J(Ut), d>
            else
                val = J;
            end
        end
    end
end
