<p align="center">
  <h2 align="center">Segmentation Guided Grasping Pipeline using Segment Anything Model(SAM) and Contact GraspNet</h2>
</p>

<img src="https://github.com/NirshalChandraSekar/Segmentation-and-Grasping/blob/cc3f69cdf154f75adbff375ed20350e29e39c3fd/image.png">

### About
We create a pipeline for Segmentation Guided Grasp generation for real-world robots. We utilize SAM by facebook (original implementation here : https://github.com/facebookresearch/segment-anything) for segementation of object and then use PyTorch implementaion of Contact GraspNet(original implementation here : https://github.com/elchun/contact_graspnet_pytorch), the original Tensorflow model of Contact GraspNet by Nvidia can be found here : https://github.com/NVlabs/contact_graspnet. This provides a method to generate grasps on objects of intresets.

### Demo
Watch the Demo Video here: https://drive.google.com/file/d/1ks-L4mX4VIew_cKRrXlJG7AtwPemp42E/view?usp=sharing

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

--> Clone this repo on your local directory, and install all the above mentioned packages. 

--> Before running the main.py file, make sure you have specified the images and the camera matrix in the main.py file if you are directly passing the images. If you are streaming from an intel realsense camera make sure the camera is connected, and change the depth scale value in the Complete_SAM_Pipeline.py file based on the model of your realsense camera.




