**1) FILE LIST**

a_Reconstruction_Capture_pair_basler.py  
b_Reconstruction_Stereo_calibrate.py  
c_Reconstruction_Disparity_Pointcloud.py  
d_Visibility_effect.py  
e1_Pointcloud_processing_Orientation.m  
e2_Pointcloud_processing_SeparatedBolt.m  
e3_Pointcloud_processing_Fitted_rectanglar.m  
e4_Pointcloud_processing_RotateBolt.m  
e5_Pointcloud_processing_Sorting.m  
f1_length_data_manual.m  
f2_boundary_map.m  
f3_pointnet-master.zip  
________________________________________
**2) PROCEDURE WORKFLOW**
Pipeline: Capture → Calibrate → Reconstruct → Process → Train/Test PointNet  
**Step A. Capture stereo images (Basler) ** 
Run: a_Reconstruction_Capture_pair_basler.py  
Goal: capture left/right images using stereo camera.  
________________________________________
**Step B. Stereo calibration (intrinsics + rectification)  **
Run: b_Reconstruction_Stereo_calibrate.py  
Input: chessboard images  
Goal: compute calibration and save parameters of cameras  
________________________________________
**Step C. Disparity → depth → point cloud**  
Run: c_Reconstruction_Disparity_Pointcloud.py  
Inputs: Captured stereo pairs (left/right) and Calibration parameters  
Outputs (common structure): disparity images and point cloud (.ply)  
________________________________________
**Step D. Visibility / occlusion effect (for CAD data)**  
Run: d_Visibility_effect.py  
Goal: apply visibility filtering (remove shadowed/hidden points) or evaluate occlusion effects  
________________________________________
**MATLAB processing pipeline (bolt point-cloud processing)  **
**Step E1.** Point cloud orientation alignment  
Run: e1_Pointcloud_processing_Orientation.m  
Goal: rotate point clouds into a consistent orientation.  

**Step E2**. Separate bolts / clusters  
Run: e2_Pointcloud_processing_SeparatedBolt.m  
Goal: cluster points and save each cluster + centroids.  

**Step E3**. Fit rectangle (bolt head geometry / theta)  
Run: e3_Pointcloud_processing_Fitted_rectanglar.m  
Goal: Estimate fitted rectangle from bolt centroids.  

**Step E4.** Rotate bolt clusters  
Run: e4_Pointcloud_processing_RotateBolt.m  
Goal: Transform the point cloud data to standard position.  

**Step E5.** Sort bolts consistently  
Run: e5_Pointcloud_processing_Sorting.m  
Goal: Sorting the bolt in bolt group.  
________________________________________
**Step F. Length/angle extraction**   
f1_length_data_manual.m  
Goal: compute loosening length and bolt angle.  
________________________________________
**PointNet Deep Learning** 
f3_pointnet-master.zip  
Goal: PointNet codebase for training/testing segmentation/classification  
