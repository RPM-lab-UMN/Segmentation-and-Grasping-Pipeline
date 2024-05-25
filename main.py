from sam.Complete_SAM_Pipeline import SAM
from cgn.contact_graspnet_pytorch.inference import CGN
import numpy as np

## Initialize the SAM class and get the input for CGN.
## Input to SAM Class is the number of segments and visualization flag.
sam = SAM(1, visualization=False)
input_for_cgn = sam.main("rs")

## Save the generated file as a npy to be used by CGN.
np.save("results/input_for_cgn.npy", input_for_cgn)

## Initialize the CGN class and get the predictions.
## Refer to Contact Grasp Net Github Page for more details on the parameters.
## Link to Original Contact GraspNet Repo : https://github.com/NVlabs/contact_graspnet
cgn = CGN(input_path="results/input_for_cgn.npy", 
          K=input_for_cgn['K'], z_range = [0.2,10],
          local_regions = True,
          filter_grasps = True,
          skip_border_objects = True,
          visualize=True, 
          forward_passes=3)

pred_grasps, grasp_scores, contact_pts, gripper_openings = cgn.inference()