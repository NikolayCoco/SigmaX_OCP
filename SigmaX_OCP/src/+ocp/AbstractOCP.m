classdef (Abstract) AbstractOCP < handle
%ABSTRACTOCP  Abstract contract for a (discretized) optimal control problem.
%
%   All the solver (gradient toolbox) ever needs to know about a specific
%   problem is expressed through this interface:
%
%       nState  / nControl   state and control dimensions
%       tau                   horizon time
%       timeGrid(N)           the time grid -> 1xN
%       initialState()        x0 -> nState x 1
%       initialControl(tg)    a (hopefully good) starting control -> nControl x N
%       costGradient(tg,U)    cost, state, costate, gradient of the cost
%                             with respect to the control samples
%       project(U)            projection of the control onto its feasible box
%       controlBounds()       lower / upper control bounds -> [Ulo, Uhi]
%
%   The physics of an individual problem is hidden behind these methods.
%   To solve a DIFFERENT optimal control problem, only subclass this class
%   and implement the methods; the solver is untouched.
%
%   Homotopy / continuation: a scalar continuation parameter is exposed as
%   `homotopyValue` (default 1, meaning "the original problem").  A subclass
%   that wants to support prolongation over a path (e.g. horizon time or a
%   terminal-cost weight) should consult `obj.homotopyValue` inside
%   costGradient / timeGrid so that the generic HomotopyContinuation wrapper
%   can drive it.  `setHomotopy` simply stores the value.

    properties (Abstract)
        nState          % state dimension
        nControl        % control dimension
        name            % short problem label
    end

    properties
        tau = 1         % horizon time (can be modified by continuation)
        homotopyValue = 1   % continuation parameter          (default 1)
        homotopyName = ''   % description of that parameter, e.g. 'tau' or 'kappa'
    end

    methods (Abstract)
        tg   = timeGrid(obj, N)                 % 1xN time grid on [0, tau]
        U    = initialControl(obj, tg)          % nControl x N starting control
        [J, X, Mu, G] = costGradient(obj, tg, U) % cost, state, costate, gradient
        Uproj = project(obj, U)                 % project control onto feasible box
        [Ulo, Uhi] = controlBounds(obj)         % box bounds, nControl x 1 each
        X0   = initialState(obj)                % nState x 1 initial condition
    end

    methods
        function obj = setHomotopy(obj, v)
        %SETHOMOTOPY  Store the current continuation parameter.
            obj.homotopyValue = v;
        end

        function printInfo(obj)
        %PRINTINFO  Short description of the problem.
            fprintf('OCP  : %s\n', obj.name);
            fprintf('state dim   = %d\n', obj.nState);
            fprintf('control dim = %d\n', obj.nControl);
            fprintf('horizon tau = %.4g\n', obj.tau);
            if ~isempty(obj.homotopyName)
                fprintf('homotopy    = %s (value %.4g)\n', obj.homotopyName, obj.homotopyValue);
            end
        end
    end
end
