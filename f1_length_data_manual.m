function [hexagon, info] = d_boundary_map_v2(X, Y, num_trials, opts)
% d_boundary_map_v2
%
% Fit a hexagon-like polygon to the 2D boundary of coplanar points (X,Y).
% Method:
%   1) Compute convex hull boundary points
%   2) Repeat num_trials:
%       - randomly pick 12 hull points
%       - form 6 lines using the same pairing logic as original code
%       - intersect consecutive lines -> 6 corners
%       - compute angle + edge-length errors
%       - keep best (angle), best (edge), best (sum)
%
% Inputs:
%   X, Y       : Nx1 numeric arrays
%   num_trials : integer number of random trials
%   opts       : optional struct
%       .seed           : rng seed for reproducibility (default: [])
%       .plot           : true/false to plot hull + best hexagon (default: false)
%       .plotErrors     : true/false to plot error evolution (default: false)
%       .detTol         : determinant tolerance for parallel lines (default: 1e-6)
%       .xlim, .ylim    : axis limits for plots (default: [-23 23])
%       .verbose        : print progress occasionally (default: false)
%       .keepErrorCurves: store error lists in info (default: true)
%
% Outputs:
%   hexagon : 6x2x3 array (corners)
%       (:,:,1) best by angle error
%       (:,:,2) best by edge-length error
%       (:,:,3) best by (angle+edge) error
%
%   info    : struct with diagnostics (best errors, indices, error curves, etc.)
%
% Notes:
% - This refactor preserves the original line-pairing logic:
%     select 12 hull indices (sorted)
%     P1 uses even positions: 2,4,6,8,10,12
%     P2 uses next odd positions: 3,5,7,9,11,1 (wrap)
%
% Dependencies:
%   convhull (base MATLAB)

if nargin < 4
    opts = struct();
end

% ----------------------------
% Defaults
% ----------------------------
opts = set_defaults(opts, struct( ...
    "seed", [], ...
    "plot", false, ...
    "plotErrors", false, ...
    "detTol", 1e-6, ...
    "xlim", [-23 23], ...
    "ylim", [-23 23], ...
    "verbose", false, ...
    "keepErrorCurves", true ...
));

hexagon = nan(6, 2, 3);

info = struct();
info.best_error1 = inf;
info.best_error2 = inf;
info.best_error3 = inf;
info.best_sel1 = [];
info.best_sel2 = [];
info.best_sel3 = [];

% ----------------------------
% Input checks
% ----------------------------
X = X(:); Y = Y(:);
valid = isfinite(X) & isfinite(Y);
X = X(valid); Y = Y(valid);

if numel(X) < 20
    warning("d_boundary_map_v2:TooFewPoints", "Too few valid points.");
    info.status = "too_few_points";
    return;
end

if ~isscalar(num_trials) || num_trials < 1
    error("num_trials must be a positive integer.");
end
num_trials = round(num_trials);

% ----------------------------
% Convex hull boundary (unique)
% ----------------------------
k = convhull(X, Y);
k = k(1:end-1); % convhull returns closed polygon (first repeats at end)
hull_points = [X(k), Y(k)];
nHull = size(hull_points, 1);

if nHull < 12
    warning("d_boundary_map_v2:TooFewHullPoints", ...
        "Convex hull has only %d points (<12). Cannot form lines as designed.", nHull);
    info.status = "too_few_hull_points";
    return;
end

% RNG once (not inside loop)
if isempty(opts.seed)
    rng("shuffle");
else
    rng(opts.seed, "twister");
end

% Optional error curves
if opts.keepErrorCurves || opts.plotErrors
    error1_list = nan(num_trials, 1);
    error2_list = nan(num_trials, 1);
else
    error1_list = [];
    error2_list = [];
end

% ----------------------------
% Main loop
% ----------------------------
for t = 1:num_trials
    if opts.verbose && mod(t, max(1, floor(num_trials/10))) == 0
        fprintf("Trial %d / %d\n", t, num_trials);
    end

    % Randomly pick 12 hull points (sorted)
    sel = sort(randperm(nHull, 12));

    % Original pairing logic (deterministic equivalent of your reshape+randi)
    p1_idx = sel(2:2:end);                      % [2 4 6 8 10 12]
    p2_idx = sel([3:2:11, 1]);                  % [3 5 7 9 11 1] wrap

    P1 = hull_points(p1_idx, :);                % 6x2
    P2 = hull_points(p2_idx, :);                % 6x2

    % Build 6 line equations: A x + B y + C = 0
    A = P1(:,2) - P2(:,2);
    B = P2(:,1) - P1(:,1);
    C = P1(:,1).*P2(:,2) - P2(:,1).*P1(:,2);
    lines = [A, B, C];

    % Intersections of consecutive lines -> 6 corners
    corners = zeros(6,2);
    parallel = false;

    for i = 1:6
        L1 = lines(i,:);
        L2 = lines(mod(i,6)+1,:);

        M = [L1(1), L1(2); L2(1), L2(2)];
        rhs = [-L1(3); -L2(3)];

        if abs(det(M)) <= opts.detTol
            parallel = true;
            break;
        end

        xy = M \ rhs;
        corners(i,:) = xy(:).';
    end

    if parallel || any(~isfinite(corners), "all")
        continue;
    end

    % Edge lengths
    edges = diff([corners; corners(1,:)], 1, 1);
    edge_lengths = vecnorm(edges, 2, 2);
    maxLen = max(edge_lengths);
    minLen = min(edge_lengths);

    if maxLen <= eps
        continue;
    end

    % Angles between consecutive edges (same as your original logic)
    angles = zeros(6,1);
    okAngles = true;

    for i = 1:6
        i1 = mod(i-1,6) + 1;
        i2 = mod(i,6) + 1;
        i3 = mod(i+1,6) + 1;

        v1 = corners(i2,:) - corners(i1,:);
        v2 = corners(i3,:) - corners(i2,:);

        denom = norm(v1) * norm(v2);
        if denom <= eps
            okAngles = false;
            break;
        end

        cos_theta = dot(v1, v2) / denom;
        cos_theta = max(-1, min(1, cos_theta));
        angles(i) = acosd(cos_theta);
    end

    if ~okAngles
        continue;
    end

    % Errors (preserve your scoring)
    error1 = (sum(abs(angles - 120) .* (angles >= 90)) + ...
              sum(abs(angles - 60)  .* (angles <  90))) / 360;

    error2 = (maxLen - minLen) / maxLen;
    error3 = error1 + error2;

    if ~isempty(error1_list)
        error1_list(t) = error1;
        error2_list(t) = error2;
    end

    % Update bests
    if error1 < info.best_error1
        info.best_error1 = error1;
        hexagon(:,:,1) = corners;
        info.best_sel1 = sel;
    end

    if error2 < info.best_error2
        info.best_error2 = error2;
        hexagon(:,:,2) = corners;
        info.best_sel2 = sel;
    end

    if error3 < info.best_error3
        info.best_error3 = error3;
        hexagon(:,:,3) = corners;
        info.best_sel3 = sel;
    end
end

% Attach curves if requested
if opts.keepErrorCurves
    info.error1_list = error1_list;
    info.error2_list = error2_list;
end

% Status
if all(isnan(hexagon), "all")
    info.status = "no_valid_hexagon_found";
    warning("d_boundary_map_v2:NoValidHex", "No valid hexagon was found. Increase num_trials or relax geometry.");
else
    info.status = "ok";
end

% Optional plots
if opts.plot
    figure("Name","Hexagon fit (hull + best sum)", "Color","w");
    hold on;
    scatter(X, Y, 10, "filled");
    plot(hull_points(:,1), hull_points(:,2), "LineWidth", 1.5);
    if ~any(isnan(hexagon(:,:,3)), "all")
        H = hexagon(:,:,3);
        plot([H(:,1); H(1,1)], [H(:,2); H(1,2)], "LineWidth", 2);
        title(sprintf("Best (error1+error2): %.4f", info.best_error3));
    else
        title("Hull (no valid hexagon found)");
    end
    axis equal;
    xlim(opts.xlim); ylim(opts.ylim);
    grid off;
    hold off;
end

if opts.plotErrors && ~isempty(error1_list)
    figure("Name","Hexagon fit errors", "Color","w");
    plot(error1_list, ".-"); hold on;
    plot(error2_list, ".-");
    xlabel("Iteration"); ylabel("Error");
    legend("Angle error","Edge-length error","Location","best");
    grid off;
    hold off;
end

end


% ============================================================
% local helper: fill defaults in opts
% ============================================================
function opts = set_defaults(opts, defaults)
f = fieldnames(defaults);
for i = 1:numel(f)
    k = f{i};
    if ~isfield(opts, k) || isempty(opts.(k))
        opts.(k) = defaults.(k);
    end
end
end
