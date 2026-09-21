classdef ControlExample < ocp.AbstractOCP
%CONTROLEXAMPLE  A second, independent optimal control problem.
%
%   This is the worked example of the descriptor (例1):
%
%       minimise  J = int_0^1 0.5 ( x^2 + u^2 ) dt
%       subject to x(0) = 10,  x' = -x^2 + u,  free terminal state
%
%   It is deliberately different from the qubit-reset problem (a scalar
%   state/control with a running cost instead of a Mayer terminal cost).
%   Because it implements the SAME AbstractOCP contract, the very same
%   GradientSolver can solve it without any change -- this is the proof that
%   the solver is an abstract tool that replaces the problem at hand.
%
%   costate: lambda' = 2 x lambda + x,  lambda(1) = 0
%   gradient: grad_u J = u - lambda

    properties
        N   = 200
        nState   = 1
        nControl = 1
        name     = 'descriptor example 1 (x''=-x^2+u)'
    end

    methods
        function obj = ControlExample(varargin)
            obj.set(varargin{:});
            obj.homotopyName = '';
        end

        function obj = set(obj, varargin)
            for i = 1:2:numel(varargin)
                obj.(varargin{i}) = varargin{i+1};
            end
        end

        function tg = timeGrid(obj, N)
            if nargin < 2, N = obj.N; end
            tg = linspace(0, obj.tau, N);
        end

        function X0 = initialState(~)
            X0 = 10;
        end

        function U = initialControl(obj, tg)
            U = zeros(1, numel(tg));
        end

        function [Ulo, Uhi] = controlBounds(~)
            Ulo = -inf;
            Uhi =  inf;
        end

        function Uproj = project(~, U)
            Uproj = U;
        end

        function [J, X, Mu, G] = costGradient(obj, tg, U)
            N = numel(tg);
            f  = @(x, u) -x.^2 + u;
            fx = @(x) -2*x;
            Lu = @(u) u;

            % forward state (RK4)
            X = zeros(1, N); X(1) = obj.initialState();
            for k = 1:N-1
                dt = tg(k+1) - tg(k);
                odef = @(x) f(x, U(k));
                X(k+1) = obj.rk4(odef, X(k), dt);
            end
            % costate (backward)
            Mu = zeros(1, N); Mu(end) = 0;   % natural BC (free terminal)
            for k = N:-1:2
                dt = tg(k) - tg(k-1);
                odef = @(la) -(obj.dlam(X(k-1), la));   % integrate backward
                Mu(k-1) = obj.rk4(odef, Mu(k), dt);
            end
            % cost and gradient
            J = trapz(tg, 0.5*(X.^2 + U.^2));
            G = U - Mu;                      % grad_u J = u - lambda
        end

        function rk = rk4(~, f, x, dt)
            k1 = f(x);
            k2 = f(x + 0.5*dt*k1);
            k3 = f(x + 0.5*dt*k2);
            k4 = f(x + dt*k3);
            rk = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
        end

        function d = dlam(~, x, lambda)
            d = 2*x*lambda + x;              % lambda' = 2 x lambda + x
        end
    end
end
