function out = c6_separated_bolt(inTxtPath, outDir, ii, opts)
% c6_separated_bolt
%
% Step (C6) of the pipeline:
% - Load a point set (Nx3) from TXT
% - Run K-means into K clusters
% - Filter points close to each centroid (radius threshold)
% - Re-run K-means on filtered points
% - Save each cluster points + a centroid summary file
%
% Typical usage:
%   opts = struct();
%   opts.save = true;
%   opts.plot = false;
%   opts.numClusters = 8;
%   opts.radiusThresh = 0.4;
%   opts.separationFactor = 2.5;  % centroid min distance > meanDistance / separationFactor
%   out = c6_separated_bolt("outputs/c1/points_rotated.txt", "outputs/c6", 1, opts);
%
% Output files:
%   outputs/c6/S1/S1_NutR_cluster1_v1.txt ... cluster8
%   outputs/c6/S1/S1_Centroid_NutR_v1.txt    (human-readable lines)
%
% Returns:
%   out struct with fields:
%     points, filteredPoints, idx, centroids, new_idx, new_centroids, outFolder

arguments
    inTxtPath (1,1) string
    outDir    (1,1) string
    ii        (1,1) double
    opts.save (1,1) logical = true
    opts.plot (1,1) logical = false

    opts.numClusters (1,1) double = 8

    % threshold(1) in your code: point-to-centroid max distance
    opts.radiusThresh (1,1) double = 0.4

    % threshold(2) in your code: meanDist / factor criterion
    opts.separationFactor (1,1) double = 2.5

    opts.maxIterations (1,1) double = 10
    opts.kmeansReplicates (1,1) double = 3
    opts.kmeansMaxIter (1,1) double = 200

    opts.versionTag (1,1) string = "v1"
end

disp("Running: c6_separated_bolt");

% -------------------------
% 1) Load points
% -------------------------
if ~isfile(inTxtPath)
    error("Input file not found: %s", inTxtPath);
end

data = load(inTxtPath);
if size(data,2) < 3
    error("Input must be Nx3 (or more columns). Got %dx%d", size(data,1), size(data,2));
end

points = data(:,1:3);

% Optional plot: raw
if opts.plot
    figure;
    scatter3(points(:,1), points(:,2), points(:,3), 1, "filled");
    xlabel("X"); ylabel("Y"); zlabel("Z");
    axis equal; grid off;
    title("C6 input points");
end

K = round(opts.numClusters);

% -------------------------
% 2) K-means (with separation check)
% -------------------------
[idx, centroids] = run_kmeans_with_separation(points, K, opts);

% -------------------------
% 3) Filter points close to each centroid
% -------------------------
colors = lines(K);
filtered_points = zeros(0,3);

if opts.plot
    figure; hold on;
end

for k = 1:K
    cluster_points = points(idx == k, :);
    if isempty(cluster_points)
        continue;
    end

    d = vecnorm(cluster_points - centroids(k,:), 2, 2);
    close_points = cluster_points(d <= opts.radiusThresh, :);

    filtered_points = [filtered_points; close_points]; %#ok<AGROW>

    if opts.plot && ~isempty(close_points)
        scatter3(close_points(:,1), close_points(:,2), close_points(:,3), 1, colors(k,:), "filled");
    end
end

if opts.plot
    scatter3(centroids(:,1), centroids(:,2), centroids(:,3), 100, "k", "x", "LineWidth", 2);
    xlabel("X"); ylabel("Y"); zlabel("Z");
    axis equal; grid off;
    title("Filtered points (near centroids)");
    hold off;
end

if size(filtered_points,1) < K
    warning("Too few filtered points (%d) for %d clusters. Consider increasing radiusThresh.", size(filtered_points,1), K);
end

% -------------------------
% 4) Re-run K-means on filtered points
% -------------------------
[new_idx, new_centroids] = run_kmeans_with_separation(filtered_points, K, opts);

% Optional plot: final clusters
if opts.plot
    figure; hold on;
    for k = 1:K
        cp = filtered_points(new_idx == k, :);
        if ~isempty(cp)
            scatter3(cp(:,1), cp(:,2), cp(:,3), 1, colors(k,:), "filled");
        end
    end
    scatter3(new_centroids(:,1), new_centroids(:,2), new_centroids(:,3), 100, "k", "x", "LineWidth", 2);
    xlabel("X"); ylabel("Y"); zlabel("Z");
    axis equal; grid off;
    title("Re-clustered filtered points");
    hold off;
end

% -------------------------
% 5) Save outputs
% -------------------------
sub_folder = fullfile(outDir, sprintf("S%d", ii));
if opts.save
    if ~exist(sub_folder, "dir"); mkdir(sub_folder); end

    % centroid summary file (human readable)
    centroid_file = fullfile(sub_folder, sprintf("S%d_Centroid_NutR_%s.txt", ii, opts.versionTag));
    fid = fopen(centroid_file, "w");
    if fid < 0
        error("Cannot open centroid file for writing: %s", centroid_file);
    end

    for k = 1:K
        cp = filtered_points(new_idx == k, :);

        % Save cluster points
        cluster_file = fullfile(sub_folder, sprintf("S%d_NutR_cluster%d_%s.txt", ii, k, opts.versionTag));
        writematrix(cp, cluster_file);

        % Save centroid line
        c = new_centroids(k,:);
        fprintf(fid, "Cluster %d: %.6f %.6f %.6f\n", k, c(1), c(2), c(3));
    end

    fclose(fid);
end

% Return struct
out = struct();
out.points = points;
out.filteredPoints = filtered_points;
out.idx = idx;
out.centroids = centroids;
out.new_idx = new_idx;
out.new_centroids = new_centroids;
out.outFolder = string(sub_folder);

end


% ========================================================================
% Helper: K-means with centroid separation check
% ========================================================================
function [idx, centroids] = run_kmeans_with_separation(points, K, opts)
max_iterations = round(opts.maxIterations);
iteration = 0;
recalculate = true;

while recalculate && iteration < max_iterations
    iteration = iteration + 1;

    % K-means settings: replicate to reduce bad local minima
    [idx, centroids] = kmeans(points, K, ...
        "Replicates", round(opts.kmeansReplicates), ...
        "MaxIter", round(opts.kmeansMaxIter));

    % Pairwise centroid distances
    dist_matrix = pdist2(centroids, centroids);
    upper_tri = triu(dist_matrix, 1);
    distances = upper_tri(upper_tri > 0);

    if isempty(distances)
        warning("Centroid distance matrix empty. (All centroids identical?) Recalculating...");
        continue;
    end

    min_dist = min(distances);
    mean_dist = mean(distances);

    fprintf("MinDist=%.4f, MeanDist=%.4f\n", min_dist, mean_dist);

    % Stop if min centroid separation is sufficiently large
    if min_dist > mean_dist / opts.separationFactor
        recalculate = false;
    else
        fprintf("Recalculating clusters: centroids too close (iter %d/%d)\n", iteration, max_iterations);
    end
end

if iteration == max_iterations
    warning("Maximum iterations reached. Clusters may still be too close.");
end
end
