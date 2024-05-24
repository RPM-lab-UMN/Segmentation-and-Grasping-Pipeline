import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import pyrealsense2 as rs
import torch

class SAM:
    def __init__(self, num_segments=1, visualization=False):
        self.output_for_cgn = None
        self.color_image = None
        self.depth_image = None
        self.k_matrix = None
        self.num_segments = num_segments
        self.visualization = visualization

    def stream_images_from_rs(self):
        print("\nCapturing the frames")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)

        align_to = rs.stream.color
        align = rs.align(align_to)

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        self.depth_image = np.asanyarray(depth_frame.get_data())
        self.color_image = np.asanyarray(color_frame.get_data())

        print("\nFrames Captured")

        depth_profile = rs.video_stream_profile(depth_frame.get_profile())
        depth_intrinsics = depth_profile.get_intrinsics()
        
        self.k_matrix = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],[0, depth_intrinsics.fy, depth_intrinsics.ppy],[0, 0, 1]])

        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        points_data = np.asarray(points.get_vertices())
        
    def direct_image_input(self, rgb, depth, k_matrix):
        self.color_image = rgb
        self.depth_image = depth
        self.k_matrix = k_matrix

    def apply_masks(self, result_dict, image_shape, original_image):
        # Create a black image of the same size as the input image
        img = np.zeros(image_shape, dtype=original_image.dtype)

        # Apply each mask to the image
        for val in result_dict:
            mask = val
            img[mask] = original_image[mask>0]  # Change pixel values where the mask is applied

        # Display the image
        return img

    def object_segmentation(self):
        r = []
        for i in range(self.num_segments):
            interest = cv2.selectROI("Interactive menu", self.color_image)
            convert = [interest[0], interest[1], interest[0]+interest[2], interest[1]+interest[3]]
            r.append(np.asarray(convert))
            cv2.destroyAllWindows()
        r = np.asarray(r)

        print("\nSegmenting the objects, this might take a while")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry["vit_h"](checkpoint="sam/sam_vit_h_4b8939.pth")
        sam.to(device)
        mask_predictor = SamPredictor(sam)
        img2 = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        mask_predictor.set_image(img2)
        masks = []

        # For each bounding box create a mask and store all the masks in a list "masks"
        for i in range(self.num_segments):
            input_box = r[i]
            masks1, _, _ = mask_predictor.predict(box=input_box, multimask_output=False)
            masks.append(masks1)

        masked_images = []
        final_mask = self.color_image
        for i in range(self.num_segments):
            output = self.apply_masks(masks[i], self.color_image.shape, self.color_image)
            masked_images.append(output)
            final_mask = cv2.subtract(final_mask, output)

        masked_images.append(final_mask)

        print("Segmentation completed")

        return masked_images, masks

    def visualize_image(self, image):
        cv2.imshow("frame", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_images(self, interface, rgb, depth, k_matrix):
        if interface == "rs":
            self.stream_images_from_rs()
        else:
            self.direct_image_input(rgb, depth, k_matrix)

    def main(self, interface, rgb_from_pb=None, depth_from_pb=None, k_matrix=None):
        self.get_images(interface, rgb_from_pb, depth_from_pb, k_matrix)
        if self.visualization:
            print("\nVisualizing the images (press 'q' to continue)")
            self.visualize_image(self.color_image)
        masked_images, masks = self.object_segmentation()
        if self.visualization:
            print("\nVisualize the segmented images (press 'q' to continue)")
            for i in range(len(masked_images)):
                self.visualize_image(masked_images[i])
        if interface == "rs":
            self.depth_image = self.depth_image*0.00025 # check the scaling factor for the camera that you are using (for L515 it is 0.00025)
        seg = np.zeros(self.depth_image.shape)
        for i,mask in enumerate(masks):
            indices = np.where(mask[0]==True)
            seg[indices]=i+1
        if self.visualization:
            self.visualize_image(seg)
        self.output_for_cgn = {"rgb": self.color_image, "depth": self.depth_image, "K": self.k_matrix, "seg": seg}
        return self.output_for_cgn

if __name__ == "__main__":
    sam = SAM()
    output_for_cgn = sam.main("rs")
    print(output_for_cgn)
