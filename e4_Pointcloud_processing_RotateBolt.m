function out = c6_rotate_bolt(folderPath, ii, opts)
% c6_rotate_bolt
%
% Step (C6): Rotate each separated cluster using a global theta (about Z-axis).
% - Reads theta (radians) from a file (written by c6_fitted_rectangle_theta)
% - Reads centroid summary file (Cluster i: x y z)
% - For each cluster file: subtract centroid, rotate around Z, (optionally scale), save
% - Writes a rotated centroid summary file (XY) for convenience
%
% Typical usage:
%   opts = struct();
%   opts.save = true; opts.plot = false;
%   opts.thetaFile = "ThetaR_S1.txt";
%   opts.centroidInFile = "S1_Centroid_NutR_v2.txt";
%   opts.clusterInPattern = "S1_NutR_cluster%d_v2.txt";
%   opts.clusterOutPattern = "S1_NutR_cluster%d_v3.txt";
%   opts.centroidOutFile = "S1_Centroid_NutR_v3.txt";
%   opts.shortSideFile = "S1_short_side_v2.txt";  % optional (for scaling)
%   out = c6_rotate_bolt("outputs/c6/S1", 1, opts);

arguments
    folderPath (1,1) string
    ii (1,1) double

    opts.save (1,1) logical = true
    opts.plot (1,1) logical = false

    opts.numClusters (1,1) double = 8

    opts.thetaFile (1,1) string = "Theta.txt"

    % input files
    opts.centroidInFile (1,1) string = ""
    opts.clusterInPattern (1,1) string = ""

    % output files
    opts.centroidOutFile (1,1) string = ""
    opts.clusterOutPattern (1,1) string = ""

    % optional scaling
    opts.applyScaling (1,1) logical = false
    opts.targetShortSide (1,1) double = 120
    opts.shortSideFile (1,1) string = ""
end

disp("Running: c6_rotate_bolt");

K = round(opts.numClusters);

% -------------------------
% Resolve default file names if user did not provide
% -------------------------
if opts.centroidInFile == ""
    opts.centroidInFile = sprintf("S%d_Centroid_NutR_v2.txt", ii);
end

if opts.centroidOutFile == ""
    opts.centroidOutFile = sprintf("S%d_Centroid_NutR_v3.txt", ii);
end

if opts.clusterInPattern == ""
    opts.clusterInPattern = sprintf("S%d_NutR_cluster%%d_v2.txt", ii);
end

if opts.clusterOutPattern == ""
    opts.clusterOutPattern = sprintf("S%d_NutR_cluster%%d_v3.txt", ii);
end

if opts.shortSideFile == ""
    opts.shortSideFile = sprintf("S%d_short_side_v2.txt", ii);
end

% -------------------------
% 1) Load theta (radians)
% -------------------------
thetaPath = fullfile(folderPath, opts.thetaFile);
if ~isfile(thetaPath)
    error("Theta file not found: %s", thetaPath);
end
theta = load(thetaPath);
theta = theta(1); % ensure scalar

Rz = [cos(theta) -sin(theta) 0;
      sin(theta)  cos(theta) 0;
      0           0          1];

% -------------------------
% 2) Load centroid XY from centroid summary
% -------------------------
centroidInPath = fullfile(folderPath, opts.centroidInFile);
if ~isfile(centroidInPath)
    error("Centroid file not found: %s", centroidInPath);
end
centroidsXY = read_centroid_xy(centroidInPath);  % Kx2

if size(centroidsXY,1) < K
    warning("Centroid file has %d clusters, expected %d.", size(centroidsXY,1), K);
end

% -------------------------
% 3) Optional scaling factor from short side
% -------------------------
scaleFactor = 1.0;
if opts.applyScaling
    shortSidePath = fullfile(folderPath, opts.shortSideFile);
    if ~isfile(shortSidePath)
        error("shortSide file not found: %s", shortSidePath);
    end
    shortSide = load(shortSidePath);
    shortSide = shortSide(1);
    scaleFactor = opts.targetShortSide / shortSide;
end

% -------------------------
% 4) Prepare centroid output writer
% -------------------------
centroidOutPath = fullfile(folderPath, opts.centroidOutFile);

fidC = -1;
if opts.save
    fidC = fopen(centroidOutPath, "w");
    if fidC < 0
        error("Cannot open centroid output: %s", centroidOutPath);
    end
end

rotatedClusters = cell(K,1);
rotatedCentroidsXY = nan(K,2);

% -------------------------
% 5) Rotate each cluster
% -------------------------
for k = 1:K
    clusterIn = fullfile(folderPath, sprintf(opts.clusterInPattern, k));
    if ~isfile(clusterIn)
        warning("Cluster file missing: %s", clusterIn);
        continue;
    end

    pts = load(clusterIn);
    if size(pts,2) < 3
        warning("Cluster %d file is not Nx3: %s", k, clusterIn);
        continue;
    end

    if k > size(centroidsXY,1)
        warning("No centroid available for cluster %d. Skipping.", k);
        continue;
    end

    centroid = [centroidsXY(k,:), 0]; % your original assumption Z=0

    % subtract centroid, rotate about Z, do NOT add centroid back (same as your code)
    pts0 = pts - centroid;
    ptsRot = (Rz * pts0')';
    ptsRot = ptsRot * scaleFactor;

    % rotate centroid itself around origin for logging (XY only)
    cRot = (Rz * centroid')';
    cRotXY = cRot(1:2)' * scaleFactor;

    rotatedClusters{k} = ptsRot;
    rotatedCentroidsXY(k,:) = cRotXY;

    if opts.save
        clusterOut = fullfile(folderPath, sprintf(opts.clusterOutPattern, k));
        writematrix(ptsRot, clusterOut);
        fprintf(fidC, "Cluster %d: %.6f %.6f\n", k, cRotXY(1), cRotXY(2));
    end

    if opts.plot
        figure; hold on;
        plot3(pts(:,1), pts(:,2), pts(:,3), "bo", "MarkerSize", 1, "MarkerFaceColor","b");
        plot3(ptsRot(:,1), ptsRot(:,2), ptsRot(:,3), "ro", "MarkerSize", 1, "MarkerFaceColor","r");
        title(sprintf("Cluster %d: Original (blue) vs Rotated (red)", k));
        xlabel("X"); ylabel("Y"); zlabel("Z");
        axis equal; grid on; hold off;
    end
end

if opts.save
    fclose(fidC);
end

% Return struct
out = struct();
out.theta = theta;
out.Rz = Rz;
out.scaleFactor = scaleFactor;
out.rotatedClusters = rotatedClusters;
out.rotatedCentroidsXY = rotatedCentroidsXY;
out.centroidOutPath = string(centroidOutPath);
out.folderPath = folderPath;

end


% ========================================================================
% Helper: read centroid XY from "Cluster i: x y z" lines
% ========================================================================
function centroidsXY = read_centroid_xy(filePath)
fid = fopen(filePath, "r");
if fid < 0
    error("Cannot open centroid file: %s", filePath);
end

centroidsXY = zeros(0,2);

while ~feof(fid)
    line = fgetl(fid);
    if ischar(line) && contains(line, "Cluster")
        vals = sscanf(line, "Cluster %d: %f %f %f");
        if numel(vals) >= 4
            centroidsXY(end+1,:) = vals(2:3)'; %#ok<AGROW>
        end
    end
end

fclose(fid);
end
