classdef Visualizer < handle
%VISUALIZER  Live 4-panel visualization + animation of the gradient iteration.
%
%   During a solve the GradientSolver calls the configured observer each
%   iteration:
%
%       v = vis.Visualizer('animate', true, 'format', 'gif', 'videoFile', 'iteration.gif');
%       s.observer = @(iter,J,X,Mu,U,tg) v.snapshot(iter,J,X,Mu,U,tg);
%       res = s.solve();
%       v.finish();
%
%   Panels: (1) state, (2) costate, (3) control, (4) cost history.
%   The number of state / control curves adapts to model.nState / nControl,
%   so the same visualizer serves both the qubit problem and the scalar example.
%
%   Animation formats:
%       'gif'   universally viewable (default)
%       'mp4'   MPEG-4 via VideoWriter (as in the original research code)

    properties
        animate  = false          % record frames to a file
        format   = 'gif'          % 'gif' | 'mp4'
        videoFile = 'iteration.gif'
        fps      = 12
        showLive = true           % draw the (visible) live figure
        xlabel   = 'time'
    end

    properties (Access = private)
        figHdl = []
        ax     = []
        Jhist  = []
        vw     = []
        gifFrames = {}
        name   = ''
    end

    methods
        function obj = Visualizer(varargin)
            obj.set(varargin{:});
        end

        function obj = set(obj, varargin)
            for i = 1:2:numel(varargin)
                obj.(varargin{i}) = varargin{i+1};
            end
        end

        function init(obj, model, tg)
        %INIT  Build the figure and open the animation sink.
            obj.name   = model.name;
            obj.Jhist  = [];
            obj.gifFrames = {};
            if obj.animate || obj.showLive
                obj.figHdl = figure('Color', 'w', ...
                                    'Position', [60 60 1200 800], ...
                                    'Name', ['iteration -- ' model.name], ...
                                    'Visible', 'on');
                for p = 1:4
                    obj.ax(p) = subplot(2, 2, p);
                    grid(obj.ax(p), 'on');
                end
                xlabel(obj.ax(1), obj.xlabel); ylabel(obj.ax(1), 'state');
                xlabel(obj.ax(2), obj.xlabel); ylabel(obj.ax(2), 'costate');
                xlabel(obj.ax(3), obj.xlabel); ylabel(obj.ax(3), 'control');
                xlabel(obj.ax(4), 'iteration'); ylabel(obj.ax(4), 'cost');
            end
            if obj.animate
                obj.openSink();
            end
        end

        function snapshot(obj, iter, J, X, Mu, U, tg)
        %SNAPSHOT  Refresh the panels and record the current iterate.
            obj.Jhist(end+1) = J;
            if obj.animate || obj.showLive
                obj.drawPanels(iter, J, X, Mu, U, tg);
                drawnow limitrate nocallbacks;
                if obj.animate
                    frame = getframe(obj.figHdl);
                    obj.writeFrame(frame);
                end
            end
        end

        function finish(obj)
        %FINISH  Close the animation sink.
            if obj.animate
                obj.closeSink();
            end
        end

        % ---------- animation sink ----------
        function openSink(obj)
            if strcmpi(obj.format, 'mp4')
                obj.vw = VideoWriter(obj.videoFile, 'MPEG-4');
                obj.vw.FrameRate = obj.fps;
                open(obj.vw);
            else
                obj.gifFrames = {};
            end
        end

        function writeFrame(obj, frame)
            if strcmpi(obj.format, 'mp4')
                writeVideo(obj.vw, frame);
            else
                obj.gifFrames{end+1} = frame.cdata;
            end
        end

        function closeSink(obj)
            if strcmpi(obj.format, 'mp4')
                if ~isempty(obj.vw), close(obj.vw); end
            elseif ~isempty(obj.gifFrames)
                obj.writeGif();
            end
        end

        function writeGif(obj)
            delay = 1/obj.fps;
            for k = 1:numel(obj.gifFrames)
                [ind, cm] = rgb2ind(obj.gifFrames{k}, 256);
                if k == 1
                    imwrite(ind, cm, obj.videoFile, 'gif', ...
                            'LoopCount', inf, 'DelayTime', delay);
                else
                    imwrite(ind, cm, obj.videoFile, 'gif', ...
                            'WriteMode', 'append', 'DelayTime', delay);
                end
            end
        end

        % ---------- drawing ----------
        function drawPanels(obj, iter, J, X, Mu, U, tg)
            ns = size(X, 1); nc = size(U, 1);
            N  = numel(tg);
            if numel(obj.Jhist) > 400, idx = 1:400; else, idx = 1:numel(obj.Jhist); end

            % state
            cla(obj.ax(1)); hold(obj.ax(1), 'on');
            for i = 1:ns
                plot(obj.ax(1), tg, X(i, :), 'LineWidth', 1.5);
            end
            xlim(obj.ax(1), [0 tg(end)]);
            legend(obj.ax(1), obj.names('p', ns), 'Location', 'best');
            hold(obj.ax(1), 'off');

            % costate
            cla(obj.ax(2)); hold(obj.ax(2), 'on');
            for i = 1:ns
                plot(obj.ax(2), tg, Mu(i, :), 'LineWidth', 1.5);
            end
            xlim(obj.ax(2), [0 tg(end)]);
            legend(obj.ax(2), obj.names('mu', ns), 'Location', 'best');
            hold(obj.ax(2), 'off');

            % control
            cla(obj.ax(3)); hold(obj.ax(3), 'on');
            for i = 1:nc
                plot(obj.ax(3), tg, U(i, :), 'LineWidth', 1.5);
            end
            xlim(obj.ax(3), [0 tg(end)]);
            legend(obj.ax(3), obj.names('u', nc), 'Location', 'best');
            hold(obj.ax(3), 'off');

            % cost history
            cla(obj.ax(4)); hold(obj.ax(4), 'on');
            semilogy(obj.ax(4), idx, obj.Jhist(idx), 'k-', 'LineWidth', 1.5);
            ylim(obj.ax(4), [min(obj.Jhist)*0.9, max(obj.Jhist)*1.1]);
            grid(obj.ax(4), 'on');
            hold(obj.ax(4), 'off');

            title(obj.ax(4), sprintf('iter %d   J = %.4e', iter, J));
            set(obj.figHdl, 'Name', sprintf('%s -- iter %d', obj.name, iter));
        end

        function L = names(~, prefix, n)
            L = cell(1, n);
            for i = 1:n
                L{i} = sprintf('%s_%d', prefix, i);
            end
        end
    end
end
