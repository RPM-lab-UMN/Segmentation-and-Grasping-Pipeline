from sam.Complete_SAM_Pipeline import SAM
from cgn.contact_graspnet_pytorch.inference import CGN
import argparse
import numpy as np

# test = np.load("/home/rpmdt05/Code/break-it/ganesh/object-disassembly/results/output_1.npy", allow_pickle=True).item()
# print("test", test)
# rgb = test['rgb']
# depth = test['depth'] 
# k_matrix = test['K']

''' input the rgb image, depth image and the camera matrix here. rgb, depth and k_matrix are numpy arrays.
    While using the depth images make sure it is scaled properly. For example the Intel Realsense L515 camera, has a scaling factor of 0.00025.'''

rgb = None
depth = None
k_matrix = None

''' call the SAM class and get the input for the CGN model. The input is saved in the results folder as input_for_cgn.npy 
    Use the interface "rs" if you are using the Intel Realsense camera, else use the interface "direct" for the direct input.'''
sam = SAM(1, visualization=False)
input_for_cgn = sam.main("rs") #, rgb, depth, k_matrix)
np.save("results/input_for_cgn.npy", input_for_cgn)

k_matrix = input_for_cgn['K']

''' call the CGN class and get the grasps, grasp scores, contact points and gripper openings '''

cgn = CGN(input_path="results/input_for_cgn.npy", K=k_matrix, z_range = [0.2,10], visualize=True, forward_passes=3)
pred_grasps, grasp_scores, contact_pts, gripper_openings = cgn.inference()