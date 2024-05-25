import glob
import os
import argparse

import transforms3d as t3d

import torch
import numpy as np
from cgn.contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from cgn.contact_graspnet_pytorch import config_utils

from cgn.contact_graspnet_pytorch.visualization_utils_o3d import visualize_grasps, show_image
from cgn.contact_graspnet_pytorch.checkpoints import CheckpointIO 
from cgn.contact_graspnet_pytorch.data import load_available_input_data

class CGN():
    def __init__(self,input_path,K=None,z_range = [0.2,10],local_regions = True,filter_grasps = True,skip_border_objects = True,visualize = False,forward_passes=1 ):
        self.global_config = config_utils.load_config('cgn/checkpoints/contact_graspnet')
        self.ckpt_dir = 'cgn/checkpoints/contact_graspnet'
        self.input_paths = input_path
        self.local_regions = local_regions
        self.filter_grasps = filter_grasps
        self.skip_border_objects = skip_border_objects
        self.z_range = z_range
        self.forward_passes = forward_passes
        self.K = K
        self.visualize = visualize

    def get_best_pose_cgn_score(self,pred_grasps, grasp_scores, contact_pts, gripper_openings):
        """Enter Your huistic here to get the best pose from the CGN predictions."""
        raise NotImplementedError

    def inference(self):
        """
        Predict 6-DoF grasp distribution for given model and input data
        
        :param global_config: config.yaml from checkpoint directory
        :param checkpoint_dir: checkpoint directory
        :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
        :param K: Camera Matrix with intrinsics to convert depth to point cloud
        :param local_regions: Crop 3D local regions around given segments. 
        :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
        :param filter_grasps: Filter and assign grasp contacts according to segmap.
        :param segmap_id: only return grasps from specified segmap_id.
        :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
        :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
        """
        # Build the model
        grasp_estimator = GraspEstimator(self.global_config)

        # Load the weights
        model_checkpoint_dir = os.path.join(self.ckpt_dir, 'checkpoints')
        checkpoint_io = CheckpointIO(checkpoint_dir=model_checkpoint_dir, model=grasp_estimator.model)
        try:
            load_dict = checkpoint_io.load('model.pt')
        except FileExistsError:
            print('No model checkpoint found')
            load_dict = {}

        
        os.makedirs('results', exist_ok=True)

        # Process example test scenes
        for p in glob.glob(self.input_paths):
            print('Loading ', p)

            pc_segments = {}
            segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(p, K=self.K)
            
            if segmap is None and (self.local_regions or self.filter_grasps):
                raise ValueError('Need segmentation map to extract local regions or filter grasps')

            if pc_full is None:
                print('Converting depth to point cloud(s)...')
                pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                                                        skip_border_objects=self.skip_border_objects, 
                                                                                        z_range=self.z_range)
            
            print(pc_full.shape)

            print('Generating Grasps...')
            pred_grasps_cam, scores, contact_pts, gripper_openings = grasp_estimator.predict_scene_grasps(pc_full, 
                                                                                        pc_segments=pc_segments, 
                                                                                        local_regions=self.local_regions, 
                                                                                        filter_grasps=self.filter_grasps, 
                                                                                        forward_passes=self.forward_passes)  
        
            # Save results
            np.savez('results/predictions_{}'.format(os.path.basename(p.replace('png','npz').replace('npy','npz'))), 
                    pc_full=pc_full, pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts, pc_colors=pc_colors, gripper_openings=gripper_openings)

            if self.visualize:
            # Visualize results          
                show_image(rgb, segmap)
                visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
            
        if not glob.glob(self.input_paths):
            print('No files found: ', self.input_paths)

        return pred_grasps_cam, scores, contact_pts, gripper_openings
