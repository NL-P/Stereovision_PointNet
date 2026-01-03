function out = c6_fitted_rectangle_theta(folderPath, opts)
% c6_fitted_rectangle_theta
%
% Step (C6): Estimate system orientation (theta) by fitting rectangles to
% centroid points (8 clusters -> choose combinations of 4).
%
% Inputs:
%   folderPath : folder containing centroid summary file (from c6_separated_bolt)
%   opts : struct with fields (recommended)
%       .save               (true/false)
%       .plot               (true/false)
%       .centroidFile       (string) e.g. "S1_Centroid_NutR_v1.txt"
%       .thetaFile          (string) e.g. "ThetaR_S1.txt"
%       .shortSideFile      (string) e.g. "short_side_S1.txt"
%       .rectErrorThresh    (double) keep/plot rectangles with error < this
%       .thetaSimilarityTol (double) relative tolerance, default 0.1
%       .thetaPostProcess   (true/false) apply: avg_theta = pi/2 - avg_theta
%
% Typical usage:
%   opts = struct();
%   opts.save = true; opts.plot = false;
%   opts.centroidFile = "S1_Centroid_NutR_v1.txt";
%   opts.thetaFile = "ThetaR_S1.txt";
%   opts.shortSideFile = "short_side_S1.txt";
%   opts.rectErrorThresh = 0.4;
%   out = c6_fitted_rectangle_theta("outputs/c6/S1", opts);

arguments
    folderPath (1,1) string
    opts.save (1,1) logical = true
    opts.plot (1,1) logical = true

    opts.centroidFile (1,1) string = "S1_Centroid_NutR_v1.txt"
    opts.thetaFile (1,1) string = "Theta.txt"
    opts.shortSideFile (1,1) string = "short_side.txt"

    opts.rectErrorThresh (1,1) double = 0.4
    opts.thetaSimilarityTol (1,1) double = 0.1
    opts.thetaPostProcess (1,1) logical = true
end

disp("Running: c6_fitted_rectangle_theta");

% -------------------------
% 1) Load centroid points (XY) from centroid summary file
% -------------------------
centroidPath = fullfile(folderPath, opts.centroidFile);
if ~isfile(centroidPath)
    error("Centroid file not found: %s", centroidPath);
end

pointsXY = read_centroid_xy(centroidPath);   % Nx2
if size(pointsXY,1) < 4
    error("Need at least 4 centroid points. Got %d.", size(pointsXY,1));
end

% -------------------------
% 2) Try all 4-point combos and fit rectangles (filter by near-rectangle check)
% -------------------------
combos = nchoosek(1:size(pointsXY,1), 4);
valid_thetas = [];   % each row returned by best_fit_rectangle()

if opts.plot
    figure; hold on;
    xlabel("X"); ylabel("Y"); axis equal;
    plot(pointsXY(:,1), pointsXY(:,2), "bo", "MarkerFaceColor","b");
    for k = 1:size(pointsXY,1)
        plot(pointsXY(k,1), pointsXY(k,2), "kx", "LineWidth",2, "MarkerSize",8);
    end
end

for i = 1:size(combos,1)
    sel = pointsXY(combos(i,:), :);

    if ~is_near_rectangle(sel)
        continue;
    end

    % NOTE:
    % best_fit_rectangle must exist in your repo.
    % It should return: [x_center, y_center, width, height, theta, error]
    [xc, yc, w, h, theta, err] = best_fit_rectangle(sel);

    valid_thetas = [valid_thetas; theta]; %#ok<AGROW>

    if opts.plot
        plot_rectangle(xc, yc, w, h, theta(1), err, opts.rectErrorThresh);
    end
end

if isempty(valid_thetas)
    warning("No valid rectangles found. Theta may be unreliable.");
    avg_theta = NaN;
    short_side = NaN;
else
    [avg_theta, short_side, chosenCol] = choose_theta(valid_thetas, opts.thetaSimilarityTol);

    if opts.thetaPostProcess
        avg_theta = (pi/2) - avg_theta; % keep your original behavior
    end

    disp("Chosen theta column: " + chosenCol);
    disp("avg_theta (rad):");
    disp(avg_theta);
end

if opts.plot
    xlim([-120 120]); ylim([-200 200]);
    hold off;
end

% -------------------------
% 3) Save outputs
% -------------------------
thetaOutPath = fullfile(folderPath, opts.thetaFile);
shortOutPath = fullfile(folderPath, opts.shortSideFile);

if opts.save
    writematrix(avg_theta, thetaOutPath);
    writematrix(short_side, shortOutPath);
end

% -------------------------
% 4) Replot rotated system (optional)
% -------------------------
rotatedXY = [];
if opts.plot && ~isnan(avg_theta)
    R = [cos(avg_theta), -sin(avg_theta); sin(avg_theta), cos(avg_theta)];
    rotatedXY = (R * pointsXY')';

    figure; hold on;
    xlabel("X"); ylabel("Y"); axis equal;
    for k = 1:size(rotatedXY,1)
        plot(rotatedXY(k,1), rotatedXY(k,2), "ro", "MarkerFaceColor","r");
    end
    hold off;
end

% Return struct
out = struct();
out.pointsXY = pointsXY;
out.valid_thetas = valid_thetas;
out.avg_theta = avg_theta;
out.short_side = short_side;
out.thetaOutPath = string(thetaOutPath);
out.shortSideOutPath = string(shortOutPath);
out.rotatedXY = rotatedXY;

end


% ========================================================================
% Helpers
% ========================================================================
function pointsXY = read_centroid_xy(filePath)
fid = fopen(filePath, "r");
if fid < 0
    error("Cannot open file: %s", filePath);
end

pointsXY = zeros(0,2);

while ~feof(fid)
    line = fgetl(fid);
    if ischar(line) && contains(line, "Cluster")
        vals = sscanf(line, "Cluster %d: %f %f %f");
        if numel(vals) >= 4
            pointsXY(end+1,:) = vals(2:3)'; %#ok<AGROW>
        end
    end
end

fclose(fid);
end


function plot_rectangle(xc, yc, w, h, theta, err, errThresh)
corners = rectangle_corners(xc, yc, w, h, theta);
corners = [corners; corners(1,:)];

if err < errThresh
    plot(corners(:,1), corners(:,2), "r", "LineWidth", 1);
else
    plot(corners(:,1), corners(:,2), "b--", "LineWidth", 0.5);
end
end


function corners = rectangle_corners(xc, yc, w, h, theta)
dx = w/2; dy = h/2;
rect = [-dx, -dy; dx, -dy; dx, dy; -dx, dy];
R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
rectRot = rect * R';
corners = rectRot + [xc, yc];
end


function [avg_theta, short_side, chosenCol] = choose_theta(valid_thetas, relTol)
% valid_thetas is Nx4 in your code logic:
% columns 1-2: theta candidates
% columns 3-4: side lengths?
%
% Your original logic:
% - make col1 <= col2 (swap) and swap corresponding side columns
% - attempt to pick column 1 or 2 based on stability of columns 3 or 4
% - if not chosen, wrap theta > pi/2 by subtracting pi, then repeat

vt = valid_thetas;

% enforce ordering col1 <= col2, keep side columns paired
for i = 1:size(vt,1)
    if vt(i,1) > vt(i,2)
        vt(i,[1 2]) = vt(i,[2 1]);
        vt(i,[3 4]) = vt(i,[4 3]);
    end
end

chosen = pick_by_side_stability(vt, relTol);

if isnan(chosen)
    % your fallback: wrap theta > pi/2
    for i = 1:size(vt,1)
        if vt(i,1) > pi/2, vt(i,1) = vt(i,1) - pi; end
        if vt(i,2) > pi/2, vt(i,2) = vt(i,2) - pi; end
        if vt(i,1) > vt(i,2)
            vt(i,[1 2]) = vt(i,[2 1]);
            vt(i,[3 4]) = vt(i,[4 3]);
        end
    end
    chosen = pick_by_side_stability(vt, relTol);
end

if isnan(chosen)
    % final fallback: use column 1
    chosen = 1;
end

avg_theta = mean(vt(:,chosen), "omitnan");
short_side = mean(vt(:,chosen+2), "omitnan"); % keep your mapping
chosenCol = chosen;
end


function chosen = pick_by_side_stability(vt, relTol)
chosen = NaN;

for kk = 3:4
    diffs = abs(vt(:,kk) - vt(1,kk));
    is_similar = all(diffs <= relTol * abs(vt(:,kk)) + 1e-12);
    if is_similar
        chosen = kk - 2; % kk=3 => chosen=1, kk=4 => chosen=2
        return;
    end
end
end
