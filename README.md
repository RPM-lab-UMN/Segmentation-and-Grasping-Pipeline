<p align="center">
  <h2 align="center">Segmentation Guided Grasping Pipeline using Segment Anything Model(SAM) and Contact GraspNet</h2>
</p>

<img src="https://github.com/NirshalChandraSekar/Segmentation-and-Grasping/blob/cc3f69cdf154f75adbff375ed20350e29e39c3fd/image.png">





### About
We have developed a pipeline for Segmentation Guided Grasp generation for real-world robots. We employ SAM by Facebook (original code here: https://github.com/facebookresearch/segment-anything) for object segmentation and PyTorch implementation of Contact GraspNet (original code here: https://github.com/elchun/contact_graspnet_pytorch). Additionally, the original Tensorflow model of Contact GraspNet by Nvidia can be found here: https://github.com/NVlabs/contact_graspnet. This methodology facilitates grasp generation on objects of interest.

### Demo
Demo Video : Speed 1.5x
https://github.com/RPM-lab-UMN/segmentation-guided-grasp-generation/assets/101336175/adb7c681-b38b-40e5-8c20-c828ead836a0

### Usage
*The Code has been tested in Python 3.9 version. With PyTorch 2.0.1 and CUDA 12.1*

##### Required Libraries/Tools
1) Contact GraspNet - Follow the steps in the official repo install all the required packages (https://github.com/elchun/contact_graspnet_pytorch)
2) Segment Anything Model 
   ```
   pip install git+https://github.com/facebookresearch/segment-anything.git
   ```
3) Realsense SDK
   ```
   pip install pyrealsense2
   ```
4) Download checkpoint for SAM from here : https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth, and move the file to ./sam

--> Clone this repo on your local directory, and install all the above mentioned packages. 

--> Before running the main.py file, make sure you have specified the images and the camera matrix in the main.py file if you are directly passing the images. If you are streaming from an intel realsense camera make sure the camera is connected, and change the depth scale value in the Complete_SAM_Pipeline.py file based on the model of your realsense camera.




