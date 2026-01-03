function out = c6_sort_clusters(folderPath, ii, opts)
% c6_sort_clusters
%
% Step (C6): Sort clusters into a stable order based on centroid positions.
% Sorting rule (same as original):
%   - Take centroids with y > 0, sort by x (ascending)
%   - Then centroids with y < 0, sort by x (ascending)
%
% Inputs:
%   folderPath : folder containing cluster files + centroid file
%   ii         : sample id used in file naming
%   opts       : struct options
%       .numClusters        (default 8)
%       .clusterInPattern   e.g. "S1_NutR_cluster%d_v3.txt"
%       .centroidInFile     e.g. "S1_Centroid_NutR_v3.txt"
%       .centroidOutFile    e.g. "S1_Centroid_NutR_v4.txt"
%       .nutOutPattern      e.g. "S1_NutR%d_v4.txt"
%       .reverseMapFile     e.g. "S1_ReverseMap.txt"
%       .save               true/false
%       .exportZeroBased    true/false (for Python mapping)
%
% Outputs:
%   out struct with:
%     sortIdx (1-based), reverseMap (0-based if exportZeroBased), sortedCentroids

arguments
    folderPath (1,1) string
    ii (1,1) double

    opts.save (1,1) logical = true
    opts.numClusters (1,1) double = 8

    opts.clusterInPattern (1,1) string = ""
    opts.centroidInFile (1,1) string = ""
    opts.centroidOutFile (1,1) string = ""
    opts.nutOutPattern (1,1) string = ""
    opts.reverseMapFile (1,1) string = ""
    opts.exportZeroBased (1,1) logical = true
end

disp("Running: c6_sort_clusters");
K = round(opts.numClusters);

% -------------------------
% Default naming if not provided
% -------------------------
if opts.clusterInPattern == ""
    opts.clusterInPattern = sprintf("S%d_NutR_cluster%%d_v3.txt", ii);
end
if opts.centroidInFile == ""
    opts.centroidInFile = sprintf("S%d_Centroid_NutR_v3.txt", ii);
end
if opts.centroidOutFile == ""
    opts.centroidOutFile = sprintf("S%d_Centroid_NutR_v4.txt", ii);
end
if opts.nutOutPattern == ""
    opts.nutOutPattern = sprintf("S%d_NutR%%d_v4.txt", ii);
end
if opts.reverseMapFile == ""
    opts.reverseMapFile = sprintf("S%d_ReverseMap.txt", ii);
end

% -------------------------
% 1) Load cluster point sets
% -------------------------
cluster_data = cell(K,1);
for k = 1:K
    f = fullfile(folderPath, sprintf(opts.clusterInPattern, k));
    if ~isfile(f)
        error("Cluster file missing: %s", f);
    end
    cluster_data{k} = load(f);
end

% -------------------------
% 2) Load centroid XY from centroid summary file
% -------------------------
centroidPath = fullfile(folderPath, opts.centroidInFile);
if ~isfile(centroidPath)
    error("Centroid file missing: %s", centroidPath);
end
centroidsXY = read_centroid_xy(centroidPath);

if size(centroidsXY,1) ~= K
    warning("Centroid count (%d) != numClusters (%d). Sorting may be inconsistent.", size(centroidsXY,1), K);
end

% -------------------------
% 3) Sorting rule: y>0 then y<0, each sorted by x
% -------------------------
centroidData = centroidsXY;

posMask = centroidData(:,2) > 0;
negMask = centroidData(:,2) < 0;

positive_y = centroidData(posMask,:);
negative_y = centroidData(negMask,:);

[~, pos_idx] = sortrows(positive_y, 1); % sort by x
[~, neg_idx] = sortrows(negative_y, 1);

sortedCentroids = [positive_y(pos_idx,:); negative_y(neg_idx,:)];

pos_indices = find(posMask);
neg_indices = find(negMask);

sortIdx = [pos_indices(pos_idx); neg_indices(neg_idx)]; % 1-based indices into original cluster list

% Export reverseMap for Python: originalIndex-1
reverseMap = sortIdx;
if opts.exportZeroBased
    reverseMap = reverseMap - 1;
end

% -------------------------
% 4) Save outputs
% -------------------------
if opts.save
    % Save reverse map (row vector)
    mapPath = fullfile(folderPath, opts.reverseMapFile);
    writematrix(reverseMap', mapPath, "Delimiter", ",");

    % Save sorted centroid summary
    centroidOutPath = fullfile(folderPath, opts.centroidOutFile);
    fidC = fopen(centroidOutPath, "w");
    if fidC < 0
        error("Cannot open centroid output: %s", centroidOutPath);
    end
    for k = 1:min(K, size(sortedCentroids,1))
        fprintf(fidC, "Cluster %d: %.6f %.6f\n", k, sortedCentroids(k,1), sortedCentroids(k,2));
    end
    fclose(fidC);

    % Re-save NutR files in sorted order as NutR1..NutR8
    for k = 1:K
        srcIdx = sortIdx(k);
        nutData = cluster_data{srcIdx};
        outFile = fullfile(folderPath, sprintf(opts.nutOutPattern, k));
        writematrix(nutData, outFile, "Delimiter", "\t");
    end
end

% Return struct
out = struct();
out.centroidsXY = centroidsXY;
out.sortedCentroids = sortedCentroids;
out.sortIdx = sortIdx;           % 1-based
out.reverseMap = reverseMap;     % 0-based if exportZeroBased
out.folderPath = folderPath;

end


% ========================================================================
% Helper: read centroid XY from "Cluster i: x y z" or "Cluster i: x y"
% ========================================================================
function centroidsXY = read_centroid_xy(filePath)
fid = fopen(filePath, "r");
if fid < 0
    error("Cannot open file: %s", filePath);
end

centroidsXY = zeros(0,2);

while ~feof(fid)
    line = fgetl(fid);
    if ischar(line) && contains(line, "Cluster")
        % Try 3D first
        vals = sscanf(line, "Cluster %d: %f %f %f");
        if numel(vals) >= 4
            centroidsXY(end+1,:) = vals(2:3)'; %#ok<AGROW>
        else
            % Try 2D format
            vals = sscanf(line, "Cluster %d: %f %f");
            if numel(vals) >= 3
                centroidsXY(end+1,:) = vals(2:3)'; %#ok<AGROW>
            end
        end
    end
end

fclose(fid);
end
