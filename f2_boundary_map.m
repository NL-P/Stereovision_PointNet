function hexagon = d_boundary_map_v2(X,Y,num_trials)
% X = data(:,1);
% Y = data(:,2);
% Z = data(:,3);

% Compute 2D convex hull (only X-Y)
k = convhull(X, Y);

% Plot 2D boundary
figure('Name', 'Hexagon', 'Position', [100 + 400*3, 100, 400, 300]);
hold on;
scatter(X, Y, 10, 'b', 'filled'); % Original points
plot(X(k), Y(k), 'r-', 'LineWidth', 2); % Convex hull boundary
xlabel('X'); ylabel('Y'); axis equal; xlim([-23 23]); ylim([-23 23]);
%title('2D Convex Hull of Coplanar Points');
grid off; 


%% Get boundary points from convex hull
hull_points = [X(k), Y(k)];
num_hull_points = length(k);

% Initialize storage for valid hexagons
hexagon = [];
% num_trials = 10000;
best_error1 = inf;
best_error2 = inf;
best_error3 = inf;

% Arrays to store errors for plotting
error1_list = nan(num_trials,1);
error2_list = nan(num_trials,1);

for ii=1:num_trials
    %if mod(ii, 1000) == 0; disp(['Iteration: ', num2str(ii)]); end
    % Randomly select 6 points for hexagon edges
    rng('shuffle');  % Ensure randomness
    selected_indices = randperm(num_hull_points, 12);  % Select 6 random indices
    selected_indices = sort(selected_indices);  % Sort to maintain order
    
    %% Divide selected points into 6 groups (each with 2 points)
    groups = reshape(selected_indices, [2, 6])';
    groups = [groups; groups(1, :)]; 
    point_1=zeros(6, 2); point_2=zeros(6, 2);
    for i=1:6
        point_1(i)=groups(i,randi([2,2]));
        point_2(i)=groups(i+1,randi([1,1]));
    end
    
    % Construct 6 lines
    lines = zeros(6, 3); % Store line equations (Ax + By + C = 0)
    parallel_detected = false; % Flag for detecting parallel lines
    for i = 1:6
        % Get two points for the current line
        P1 = hull_points(point_1(i), :);
        P2 = hull_points(point_2(i), :);
    
        % Compute line equation: Ax + By + C = 0
        A = P1(2) - P2(2);
        B = P2(1) - P1(1);
        C = P1(1) * P2(2) - P2(1) * P1(2);
        lines(i, :) = [A, B, C];
    end
    
    % Find intersection points of consecutive lines
    hexagon_corners = zeros(6, 2);
    for i = 1:6
        % Get two lines
        L1 = lines(i, :);
        L2 = lines(mod(i,6) + 1, :); % Wraps around at the end
    
        % Solve for intersection (Ax + By = -C)
        A = [L1(1), L1(2); L2(1), L2(2)];
        C = [-L1(3); -L2(3)];
    
        if abs(det(A)) > 1e-6  % Ensure lines are not parallel
            hexagon_corners(i, :) = (A \ C)'; % Solve for [x, y]
        else
            parallel_detected = true;
            break;
        end
    end

    if parallel_detected
        continue; % Move to next ii iteration
    end
    %%
    % Compute edge lengths
    edge_lengths = vecnorm(diff([hexagon_corners; hexagon_corners(1, :)]), 2, 2);
    
    % Compute angles between edges
    angles = zeros(6, 1);
    
    for i = 1:6
        % Get vectors for current and next edge
        v1 = hexagon_corners(mod(i,6)+1,:) - hexagon_corners(i,:);
        v2 = hexagon_corners(mod(i+1,6)+1,:) - hexagon_corners(mod(i,6)+1,:);
        
        % Compute angle in degrees
        cos_theta = dot(v1, v2) / (norm(v1) * norm(v2));
        angles(i) = acosd(max(-1, min(1, cos_theta)));  % Ensure valid range
    end
    
    % Calculate errors
    error1 = (sum(abs(angles - 120) .* (angles >= 90)) + ...
             sum(abs(angles - 60)  .* (angles < 90)))/360;
    
    max_length = max(edge_lengths);
    min_length = min(edge_lengths);

    error2 = (max_length - min_length)/max_length; % Edge length variation

    % Save errors
    error1_list(ii) = error1;
    error2_list(ii) = error2;

    % Check for minimum error1
    if error1 < best_error1
        best_error1 = error1;
        hexagon(:,:,1) = hexagon_corners;
    end

    % Check for minimum error2
    if error2 < best_error2
        best_error2 = error2;
        hexagon(:,:,2) = hexagon_corners;
    end

    if error1+error2 < best_error3
        best_error3 = error1+error2;
        hexagon(:,:,3) = hexagon_corners;
    end
end


%% Plot error1 and error2
figure
plot(error1_list,'b.-','LineWidth',0.5); hold on;
plot(error2_list, '.-', 'Color', [0.85, 0.33, 0.10], 'LineWidth', 0.5);
xlabel('Iteration'); ylabel('Error Value');
legend('Error of angles','Error of edge lengths','Location','best');
%title('Error evolution during hexagon fitting');
ylim([0 1.2])
grid off;


%% Plot the first valid hexagon by Angle Check
% figure; hold on;
% plot(X, Y, 'bo', 'MarkerSize', 3); % Original points
% plot(hull_points(:,1), hull_points(:,2), 'g-', 'LineWidth', 1); % Convex Hull
% plot(hexagon(:,1,1), hexagon(:,2,1), 'ro-', 'MarkerSize', 6, 'LineWidth', 2); % First valid hexagon
% plot([hexagon(:,1,1); hexagon(1,1,1)], ...
%      [hexagon(:,2,1); hexagon(1,2,1)], 'r-', 'LineWidth', 2); % Close hexagon
% xlabel('X-axis'); ylabel('Y-axis');
% title('Valid Hexagon Detection from Convex Hull by Angle Check');
% grid on; axis equal;
% legend({'Data Points', 'Convex Hull', 'Hexagon Edges'}, 'Location', 'best');
% 
% % Plot the first valid hexagon by Length Check
% figure; hold on;
% plot(X, Y, 'bo', 'MarkerSize', 3); % Original points
% plot(hull_points(:,1), hull_points(:,2), 'g-', 'LineWidth', 1); % Convex Hull
% plot(hexagon(:,1,2), hexagon(:,2,2), 'ro-', 'MarkerSize', 6, 'LineWidth', 2); % First valid hexagon
% plot([hexagon(:,1,2); hexagon(1,1,2)], ...
%      [hexagon(:,2,2); hexagon(1,2,2)], 'r-', 'LineWidth', 2); % Close hexagon
% xlabel('X-axis'); ylabel('Y-axis');
% title('Valid Hexagon Detection from Convex Hull by Length Check');
% grid on; axis equal;
% legend({'Data Points', 'Convex Hull', 'Hexagon Edges'}, 'Location', 'best');
% 
% % Plot the first valid hexagon by Length Check + Angle Check
% figure; hold on;
% plot(X, Y, 'bo', 'MarkerSize', 3); % Original points
% plot(hull_points(:,1), hull_points(:,2), 'g-', 'LineWidth', 1); % Convex Hull
% plot(hexagon(:,1,3), hexagon(:,2,3), 'ro-', 'MarkerSize', 6, 'LineWidth', 2); % First valid hexagon
% plot([hexagon(:,1,3); hexagon(1,1,3)], ...
%      [hexagon(:,2,3); hexagon(1,2,3)], 'r-', 'LineWidth', 2); % Close hexagon
% xlabel('X-axis'); ylabel('Y-axis');
% title('Valid Hexagon Detection from Convex Hull by Length Check + Angle Check');
% grid on; axis equal;
% legend({'Data Points', 'Convex Hull', 'Hexagon Edges'}, 'Location', 'best');

end
