classdef QubitReset < ocp.AbstractOCP
%QUBITRESET  Time-optimal qubit-reset problem driven by a sigma_x/sigma_z drive.
%
%   This is the concrete optimal control problem that the gradient toolbox
%   is applied to.  It implements the AbstractOCP contract, so the solver is
%   completely decoupled from the physics.
%
%   Model (Lindblad master equation of a qubit subject to a dissipative
%   bath with a Lorentzian spectral density, dephasing rate gamma, cut-off
%   frequency omega_c):
%
%       control u(t) = [lambda_x(t); lambda_z(t)]  (drive amplitudes)
%       theta(t) = atan2(lambda_x, lambda_z),  Lambda(t) = |u(t)|
%       state  p(t) = [p_e; p_r; p_i]           (3 x N)
%       d p / dt = P(theta,Lambda) p + p0(theta,Lambda)
%
%   Cost:  J = |p(tau)|          (sigma_x-reset Mayer term)
%          optionally  J = kappa |p(tau)| + (1-kappa)/tau int (u/lambda_max) dt
%          where the second (regularizing) form is used during a homotopy on
%          the continuation weight kappa (see HomotopyContinuation).
%
%   Path constraint: 0 <= lambda_x <= lambda_x_max, 0 <= lambda_z <= lambda_z_max,
%   enforced by PROJECTION (more stable than the penalty function; the penalty
%   approach is discussed in the descriptor).
%
%   Using this class as a model only requires its physics.  To solve a
%   different problem, subclass AbstractOCP instead.

    properties
        omega_c      % spectral-density cut-off frequency
        gamma        % dephasing / relaxation rate
        lambdaMax    % 2 x 1 upper control bounds
        pInitial     % 3 x 1 initial density matrix population
        N            % number of control samples (default 100)
        nState   = 3
        nControl = 2
        name     = 'time-optimal qubit reset (sigma_x)'
    end

    methods
        function obj = QubitReset(varargin)
        %QUBITRESET  Construct with name-value options.
        %   QubitReset('omega_c',1,'gamma',10,'lambdaMax',[3;2],'tau',0.15,'N',100)
            obj.omega_c   = 1;
            obj.gamma     = 10;
            obj.lambdaMax = [3; 2];
            obj.pInitial  = [0.5; 0; 0];
            obj.tau       = 0.15;
            obj.N         = 100;
            obj.homotopyName = 'kappa';   % continuation weight (0 -> 1)
            obj.homotopyValue = 1;
            obj.set(varargin{:});
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

        function X0 = initialState(obj)
        %INITIALSTATE  Initial state p(0).
        %   NOTE (fixed): the original hard-coded [0.5;0;0] and ignored pInitial,
        %   while costGradient integrates forward using pInitial. The two therefore
        %   disagreed as soon as pInitial was changed. Now it returns pInitial,
        %   whose default IS [0.5;0;0] -- so default-case numerics are unchanged.
            X0 = obj.pInitial;
        end

        function U = initialControl(obj, tg)
            n = numel(tg);
            b = obj.lambdaMax;
            % parabolic ramp (physically motivated warm start)
            ux = 4*b(1)/obj.tau^2 * tg .* (obj.tau - tg);
            uz1 = 4*b(2)/obj.tau^2 * tg .* (obj.tau - tg);
            uz2 = b(2)*ones(1, n);
            uz = [uz1(1:round(n/2)) uz2(round(n/2)+1:end)];
            U = [ux; uz];
        end

        function [Ulo, Uhi] = controlBounds(obj)
            Ulo = zeros(2, 1);
            Uhi = obj.lambdaMax;
        end

        function Uproj = project(obj, U)
            [lo, hi] = obj.controlBounds();
            Uproj = max(lo, min(hi, U));
        end

        function [J, X, Mu, G] = costGradient(obj, tg, U)
        %COSTGRADIENT  Cost, state, costate and cost gradient of the problem.
            [theta, Lambda] = obj.controlTransform(U);
            N = numel(tg);
            kappa = obj.homotopyValue;

            X = obj.forwardIntegrate(tg, theta, Lambda, obj.pInitial);

            pT = X(:, end);
            nrm = sqrt(sum(pT.^2));
            if nrm < 1e-14, nrm = 1e-14; end
            J_ter = nrm;
            mu0 = -kappa * pT / nrm;                    % natural boundary condition
            Mu = obj.backwardIntegrate(tg, theta, Lambda, mu0);

            G = -obj.gradientAt(X, Mu, theta, Lambda);  % G = grad_u J  (note the sign)

            if kappa < 1
                % regularizing running cost (homotopy on kappa)
                Lr = (1-kappa)/obj.tau .* sum(U ./ obj.lambdaMax, 1);
                J  = kappa*J_ter + trapz(tg, Lr);
                dLr = (1-kappa)/obj.tau ./ obj.lambdaMax;     % nControl x 1
                G   = G + dLr;
            else
                J = J_ter;
            end
        end

        function obj = setHomotopy(obj, v)
        %SETHOMOTOPY  Dispatch the continuation parameter.
        %   'tau'   -> change the horizon (prolongation in time)
        %   'kappa' -> change the terminal-cost weight / regularization
            if strcmpi(obj.homotopyName, 'tau')
                obj.tau = v;
            end
            obj.homotopyValue = v;
        end

        function res = solveReference(obj)
        %SOLVEREFERENCE  Wrapper that solves the problem using the default solver.
            s = solver.GradientSolver(obj);
            s.direction = 'gradient'; s.lineSearch = 'armijo';
            res = s.solve(obj.initialControl(obj.timeGrid(obj.N)));
        end
    end

    methods (Access = protected)
        % ---------------- control transforms ----------------
        function [theta, Lambda] = controlTransform(~, U)
            ux = U(1, :); uz = U(2, :);
            theta  = atan2(ux, uz);
            Lambda = sqrt(ux.^2 + uz.^2);
        end

        function [tx, tz, lx, lz] = transformJacobian(~, theta, Lambda)
            invL = 1./max(Lambda, 1e-12);
            tx = cos(theta) .* invL;    % d theta / d lambda_x  (lambda_z = Lambda cos)
            tz = -sin(theta) .* invL;   % d theta / d lambda_z  (lambda_x = Lambda sin)
            lx = sin(theta);             % d Lambda / d lambda_x
            lz = cos(theta);             % d Lambda / d lambda_z
            % polar coordinates are singular at Lambda = 0 (matches the
            % original derivation: the Jacobian is killed there to avoid 0/0)
            small = Lambda < 1e-9;
            tx(small) = 0; tz(small) = 0;
        end

        % ---------------- integrators (RK4, piecewise-constant control) ----------------
        function X = forwardIntegrate(obj, tg, theta, Lambda, X0)
            N = numel(tg);
            X = zeros(obj.nState, N);
            X(:, 1) = X0;
            for k = 1:N-1
                dt = tg(k+1) - tg(k);
                [A, b] = obj.matVec(theta(k), Lambda(k));
                f = @(x) A*x + b;
                X(:, k+1) = obj.rk4(f, X(:, k), dt);
            end
        end

        function Mu = backwardIntegrate(obj, tg, theta, Lambda, Mu0)
            N = numel(tg);
            Mu = zeros(obj.nState, N);
            Mu(:, N) = Mu0;
            for k = N:-1:2
                dt = tg(k) - tg(k-1);
                [A, ~] = obj.matVec(theta(k-1), Lambda(k-1));
                f = @(mu) A'*mu;               % d mu / d s with s = -t   (integrate forward in s)
                Mu(:, k-1) = obj.rk4(f, Mu(:, k), dt);
            end
        end

        function x = rk4(~, f, x, dt)
            k1 = f(x);
            k2 = f(x + 0.5*dt*k1);
            k3 = f(x + 0.5*dt*k2);
            k4 = f(x + dt*k3);
            x  = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
        end

        % ---------------- gradient ----------------
        function G = gradientAt(obj, X, Mu, theta, Lambda)
            N = numel(theta);
            [Pth, p0th] = obj.matVecTheta(theta, Lambda);   % 3x3xN, 3xN
            [Pla, p0la] = obj.matVecLambda(theta, Lambda);
            Hth = zeros(1, N); Hla = zeros(1, N);
            for k = 1:N
                Hth(k) = Mu(:, k)'*(Pth(:,:,k)*X(:, k) + p0th(:, k));
                Hla(k) = Mu(:, k)'*(Pla(:,:,k)*X(:, k) + p0la(:, k));
            end
            [tx, tz, lx, lz] = obj.transformJacobian(theta, Lambda);
            G = [ tx .* Hth + lx .* Hla ;
                  tz .* Hth + lz .* Hla ];
        end
    end

    % ===================================================================
    %   Physics: all functions of the control transform (theta, Lambda).
    %   The expressions are transcribed from the analytical Derivation
    %   (see descriptor).  The convention is
    %       d p/dt = P(theta,Lambda) p + p0(theta,Lambda).
    % ===================================================================
    methods (Access = protected)
        function re = spec(obj, omega)       % spectral density J(omega)
            wc = obj.omega_c;
            re = pi*omega.*wc.^2 ./ (omega.^2 + wc.^2);
        end

        function re = dspec(obj, omega)      % d J / d omega
            wc = obj.omega_c;
            re = pi*wc.^2 .* (wc.^2 - omega.^2) ./ (omega.^2 + wc.^2).^2;
        end

        function re = Gz(~, th, la)          % Gamma_z
            re = pi*sin(th).^2;
        end
        function re = Gz_th(~, th, la)
            re = 2*pi*sin(th).*cos(th);
        end
        function re = Gz_la(~, th, la)
            re = 0*th;
        end

        function re = Gp(obj, th, la)        % Gamma_+
            re = 0.5*obj.spec(2*la).*(1./tanh(la)+1).*cos(th).^2;
            re(isnan(re))=0;
            re = re + (la==0).*(0.5*pi*cos(th).^2);
        end
        function re = Gp_th(obj, th, la)
            re = -obj.spec(2*la).*(1./tanh(la)+1).*cos(th).*sin(th);
            re(isnan(re))=0;
            re = re + (la==0).*(-pi*cos(th).*sin(th));
        end
        function re = Gp_la(obj, th, la)
            re = (obj.dspec(2*la).*(1./tanh(la)+1) - 0.5*obj.spec(2*la).*(1./sinh(la)).^2).*cos(th).^2;
            re(isnan(re))=0;
            re = re + (la==0).*(obj.dspec(0).*cos(th).^2);
        end

        function re = Gm(obj, th, la)        % Gamma_-
            re = 0.5*obj.spec(2*la).*(1./tanh(la)-1).*cos(th).^2;
            re(isnan(re))=0;
            re = re + (la==0).*(0.5*pi*cos(th).^2);
        end
        function re = Gm_th(obj, th, la)
            re = -obj.spec(2*la).*(1./tanh(la)-1).*cos(th).*sin(th);
            re(isnan(re))=0;
            re = re + (la==0).*(-pi*cos(th).*sin(th));
        end
        function re = Gm_la(obj, th, la)
            re = (obj.dspec(2*la).*(1./tanh(la)-1) - 0.5*obj.spec(2*la).*(1./sinh(la)).^2).*cos(th).^2;
            re(isnan(re))=0;
            re = re + (la==0).*(-obj.dspec(0).*cos(th).^2);
        end

        function [A, b] = matVec(obj, th, la)      % P, p0 at one sample
            g  = obj.gamma;
            Gz = obj.Gz(th, la);
            Gp = obj.Gp(th, la);
            Gm = obj.Gm(th, la);
            c  = cos(th); s = sin(th);
            A = zeros(3, 3);
            A(1,1) = -g*(4*Gz*s.^2 + (Gp+Gm)*(1+c.^2));
            A(1,2) =  g*(4*Gz - Gp - Gm)*c.*s;
            A(1,3) = -2*la.*s;
            A(2,1) =  g*(4*Gz - Gp - Gm)*c.*s;
            A(2,2) = -g*(4*Gz*c.^2 + (Gp+Gm)*(1+s.^2));
            A(2,3) =  2*la.*c;
            A(3,1) =  2*la.*s;
            A(3,2) = -2*la.*c;
            A(3,3) = -g*(4*Gz + Gp + Gm);
            b = [ g*(2*Gz*s.^2 + 0.5*(Gp*(1-c).^2 + Gm*(1+c).^2));
                 -g*(2*Gz*c.*s - 0.5*(Gp*s.*(c-2) + Gm*s.*(c+2)));
                 -la.*s ];
            A(isnan(A)) = 0; b(isnan(b)) = 0;
        end

        function [A, b] = matVecTheta(obj, th, la)  % d P/d theta, d p0/d theta
            g   = obj.gamma;
            Gz  = obj.Gz(th, la);    Gzt = obj.Gz_th(th, la);
            Gp  = obj.Gp(th, la);    Gpt = obj.Gp_th(th, la);
            Gm  = obj.Gm(th, la);    Gmt = obj.Gm_th(th, la);
            c = cos(th); s = sin(th);
            A = zeros(3, 3, numel(th));
            A(1,1,:) = -g*(4*Gzt.*s.^2 + (Gpt+Gmt).*(1+c.^2)) ...
                       -g*(4*Gz.*2.*s.*c + (Gp+Gm).*(-2.*s.*c));
            A(1,2,:) =  g*(4*Gzt-Gpt-Gmt).*c.*s + g*(4*Gz-Gp-Gm).*(c.^2-s.^2);
            A(1,3,:) = -2*la.*c;
            A(2,1,:) =  g*(4*Gzt-Gpt-Gmt).*c.*s + g*(4*Gz-Gp-Gm).*(c.^2-s.^2);
            A(2,2,:) = -g*(4*Gzt.*c.^2 + (Gpt+Gmt).*(1+s.^2)) ...
                       -g*(4*Gz.*(-2.*s.*c) + (Gp+Gm).*(2.*s.*c));
            A(2,3,:) = -2*la.*s;
            A(3,1,:) =  2*la.*c;
            A(3,2,:) =  2*la.*s;
            A(3,3,:) = -g*(4*Gzt+Gpt+Gmt);
            % NOTE (fixed): d/dth[sin(th)(cos(th)-2)] = cos^2 - 2cos - sin^2,
            % so the Gp factor must be (c^2-2c-s^2). The original wrote
            % (s^2-c^2-2c), which differs by -2*(c^2-s^2).
            % The Gm factor (c^2-s^2+2c) happens to equal the correct
            % (c^2+2c-s^2) by commutativity; both are written in symmetric
            % form below so the pair is easier to check at a glance.
            b = [ g*(2*Gzt.*s.^2 + 0.5*(Gpt.*(1-c).^2 + Gmt.*(1+c).^2)) ...
                + g*(2*Gz.*(2.*s.*c) + 0.5*(Gp.*2.*(1-c).*s + Gm.*2.*(1+c).*(-s)));
                 -g*(2*Gzt.*c.*s - 0.5*(Gpt.*s.*(c-2) + Gmt.*s.*(c+2))) ...
                - g*(2*Gz.*(c.^2-s.^2) - 0.5*(Gp.*(c.^2-2.*c-s.^2) + Gm.*(c.^2+2.*c-s.^2)));
                 -la.*c ];
            A(isnan(A)) = 0; b(isnan(b)) = 0;
        end

        function [A, b] = matVecLambda(obj, th, la) % d P/d Lambda, d p0/d Lambda
            g   = obj.gamma;
            Gz  = obj.Gz(th, la);    Gzl = obj.Gz_la(th, la);
            Gp  = obj.Gp(th, la);    Gpl = obj.Gp_la(th, la);
            Gm  = obj.Gm(th, la);    Gml = obj.Gm_la(th, la);
            c = cos(th); s = sin(th);
            A = zeros(3, 3, numel(th));
            A(1,1,:) = -g*(4*Gzl.*s.^2 + (Gpl+Gml).*(1+c.^2));
            A(1,2,:) =  g*(4*Gzl-Gpl-Gml).*c.*s;
            A(1,3,:) = -2.*s;
            A(2,1,:) =  g*(4*Gzl-Gpl-Gml).*c.*s;
            A(2,2,:) = -g*(4*Gzl.*c.^2 + (Gpl+Gml).*(1+s.^2));
            A(2,3,:) =  2.*c;
            A(3,1,:) =  2.*s;
            A(3,2,:) = -2.*c;
            A(3,3,:) = -g*(4*Gzl+Gpl+Gml);
            b = [ g*(2*Gzl.*s.^2 + 0.5*(Gpl.*(1-c).^2 + Gml.*(1+c).^2));
                 -g*(2*Gzl.*c.*s - 0.5*(Gpl.*s.*(c-2) + Gml.*s.*(c+2)));
                 -s ];
            A(isnan(A)) = 0; b(isnan(b)) = 0;
        end
    end
end
