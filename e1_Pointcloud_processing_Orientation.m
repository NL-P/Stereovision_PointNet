function out = c1_normal_rotate_pointcloud(plyPath, outDir, opts)
% c1_normal_rotate_pointcloud
%
% Step (C1) of the pointcloud processing pipeline:
% 1) Load a point cloud from a .ply file
% 2) Center it (subtract mean)
% 3) Estimate dominant plane normal by sampling many random triangles
% 4) Rotate so the dominant normal aligns with +Z (Rodrigues)
% 5) Optionally flip if upside-down
% 6) Optionally shift Z so the "dominant" Z level becomes 0 (+offset)
% 7) Save results to outDir
%
% Typical usage:
%   opts = struct();
%   opts.save = true;
%   opts.plot = false;
%   opts.nSamples = 50000;
%   opts.roundNormal = 5;
%   opts.roundZ = 3;
%   opts.zOffset = 0.06;
%   opts.outPointsName = "points_rotated.txt";
%   opts.outNormalName = "dominant_normal.txt";
%   opts.outMeanName = "centroid_mean.txt";
%   out = c1_normal_rotate_pointcloud("data/pc/frame001.ply", "outputs/c1", opts);
%
% Repo note:
% - Input data (PLY) typically lives under data/ (gitignored if large).
% - Outputs go to outputs/ (gitignored).
%
% Returns:
%   out struct with fields:
%     pointsCentered, pointsRotated, meanXYZ, dominantNormal, R, zShift

arguments
    plyPath (1,1) string
    outDir  (1,1) string
    opts.save (1,1) logical = true
    opts.plot (1,1) logical = false
    opts.nSamples (1,1) double = 50000          % number of random triangle normals
    opts.sampleRatio (1,1) double = 1.0         % 1.0 means use all points; <1 to subsample
    opts.roundNormal (1,1) double = 5           % decimals for grouping normals
    opts.roundZ (1,1) double = 3                % decimals for grouping Z for mode
    opts.zOffset (1,1) double = 0.06            % after shifting by dominant Z
    opts.flipIfUpsideDown (1,1) logical = true
    opts.outPointsName (1,1) string = "points_rotated.txt"
    opts.outNormalName (1,1) string = "dominant_normal.txt"
    opts.outMeanName (1,1) string = "centroid_mean.txt"
end

disp("Running: c1_normal_rotate_pointcloud");

% -------------------------
% 1) Load point cloud
% -------------------------
pc = pcread(plyPath);
pts = pc.Location;

if isempty(pts) || size(pts,2) < 3
    error("PLY has no valid XYZ points: %s", plyPath);
end

x = pts(:,1); y = pts(:,2); z = pts(:,3);

% -------------------------
% 2) Center
% -------------------------
meanX = mean(x, "omitnan");
meanY = mean(y, "omitnan");
meanZ = mean(z, "omitnan");
meanXYZ = [meanX, meanY, meanZ];

points = [x-meanX, y-meanY, z-meanZ];

% Optional plot (initial)
if opts.plot
    figure; scatter3(points(:,1), points(:,2), points(:,3), 1, "filled");
    xlabel("X"); ylabel("Y"); zlabel("Z");
    title("Initial centered point cloud");
    axis equal; grid on; view([70 30]);
end

% -------------------------
% 3) Subsample points to reduce cost
% -------------------------
N = size(points,1);
useN = max(3, round(N * opts.sampleRatio));
if useN < N
    idx = randperm(N, useN);
    pointsSel = points(idx,:);
else
    pointsSel = points;
end

% -------------------------
% 4) Estimate dominant normal via random triangles
% -------------------------
nSamples = round(opts.nSamples);
normalVectors = zeros(nSamples, 3);

for i = 1:nSamples
    triIdx = randperm(size(pointsSel,1), 3);
    p1 = pointsSel(triIdx(1),:);
    p2 = pointsSel(triIdx(2),:);
    p3 = pointsSel(triIdx(3),:);

    v1 = p2 - p1;
    v2 = p3 - p1;

    n = cross(v1, v2);
    nn = norm(n);

    if nn < 1e-12 || any(~isfinite(n))
        normalVectors(i,:) = [0 0 0];
    else
        normalVectors(i,:) = n / nn;
    end
end

% Remove invalid normals
validMask = all(isfinite(normalVectors),2) & vecnorm(normalVectors,2,2) > 0;
normalVectors = normalVectors(validMask,:);

if isempty(normalVectors)
    error("Could not estimate normals. Check your point cloud quality.");
end

% Optional histograms
if opts.plot
    figure;
    subplot(1,3,1); histogram(normalVectors(:,1), 40, "Normalization","pdf"); xlabel("X"); ylabel("Density"); grid off;
    subplot(1,3,2); histogram(normalVectors(:,2), 40, "Normalization","pdf"); xlabel("Y"); ylabel("Density"); grid off;
    subplot(1,3,3); histogram(normalVectors(:,3), 40, "Normalization","pdf"); xlabel("Z"); ylabel("Density"); grid off;
    disp("Normal component distributions plotted.");
end

% Mode of normals (rounded grouping)
nr = round(normalVectors, opts.roundNormal);
[uniqueNormals, ~, ic] = unique(nr, "rows");
counts = accumarray(ic, 1);
[~, mx] = max(counts);
dominantNormal = uniqueNormals(mx,:);

disp("Dominant normal (rounded mode):");
disp(dominantNormal);

% If already aligned with Z (or -Z), skip rotation
zAxis = [0 0 1];
if isequal(dominantNormal, [0 0 1]) || isequal(dominantNormal, [0 0 -1])
    R = eye(3);
    pointsRot = pointsSel;
else
    % -------------------------
    % 5) Rodrigues rotation: n1 -> +Z
    % -------------------------
    n1 = dominantNormal / norm(dominantNormal);
    n2 = zAxis;

    rotAxis = cross(n1, n2);
    rotAxisNorm = norm(rotAxis);

    if rotAxisNorm < 1e-12
        R = eye(3); % already aligned
    else
        rotAxis = rotAxis / rotAxisNorm;
        cosTheta = max(-1, min(1, dot(n1,n2)));
        theta = acos(cosTheta);

        K = [ 0        -rotAxis(3)  rotAxis(2);
              rotAxis(3) 0         -rotAxis(1);
             -rotAxis(2) rotAxis(1) 0         ];

        R = eye(3) + sin(theta)*K + (1-cos(theta))*(K*K);
    end

    pointsRot = (R * pointsSel')';
end

% -------------------------
% 6) Flip if upside-down (your logic)
% -------------------------
if opts.flipIfUpsideDown
    if abs(max(pointsRot(:,3))) > abs(min(pointsRot(:,3)))
        pointsRot(:,2) = -pointsRot(:,2);
        pointsRot(:,3) = -pointsRot(:,3);
    end
end

% -------------------------
% 7) Shift Z by dominant Z "mode" (your idea)
% -------------------------
zRot = pointsRot(:,3);
zRounded = round(zRot, opts.roundZ);
[uniqZ, ~, iz] = unique(zRounded);
cz = accumarray(iz, 1);
[~, miz] = max(cz);
dominantZ = uniqZ(miz);

pointsRot(:,3) = pointsRot(:,3) - dominantZ + opts.zOffset;

% Optional plots
if opts.plot
    figure; scatter3(pointsRot(:,1), pointsRot(:,2), pointsRot(:,3), 1, "filled");
    xlabel("X"); ylabel("Y"); zlabel("Z");
    title("Rotated + shifted point cloud");
    axis equal; grid on; view([70 30]);

    figure; scatter3(pointsRot(:,1), pointsRot(:,2), pointsRot(:,3), 1, "filled");
    xlabel("X"); ylabel("Y"); zlabel("Z");
    title("View 1"); axis equal; grid on; view([90 0]);

    figure; scatter3(pointsRot(:,1), pointsRot(:,2), pointsRot(:,3), 1, "filled");
    xlabel("X"); ylabel("Y"); zlabel("Z");
    title("View 2"); axis equal; grid on; view([0 0]);
end

% -------------------------
% 8) Save outputs
% -------------------------
outDir = string(outDir);
if opts.save
    if ~exist(outDir, "dir"); mkdir(outDir); end

    writematrix(dominantNormal, fullfile(outDir, opts.outNormalName), "Delimiter","tab");
    writematrix(meanXYZ,        fullfile(outDir, opts.outMeanName),   "Delimiter","tab");
    writematrix(pointsRot,      fullfile(outDir, opts.outPointsName));
end

% Package outputs
out = struct();
out.pointsCentered = points;
out.pointsRotated = pointsRot;
out.meanXYZ = meanXYZ;
out.dominantNormal = dominantNormal;
out.R = R;
out.zShift = dominantZ;
out.source = plyPath;
out.outDir = outDir;

end
