"""
ADAPTED THIS SCRIPT FROM DGX TO TRITON1 SETUP
"""
import sys
import json
import os, random
import cv2
import numpy as np
import copy
from PIL import Image
from descriptors.dense_correspondence_network import DenseCorrespondenceNetwork
from descriptors.find_correspondences import CorrespondenceFinder 
#from sklearn.neighbors import NearestNeighbors
from itertools import product

COLOR_RED = np.array([0, 0, 255])
COLOR_GREEN = np.array([0,255,0])

#utils.set_cuda_visible_devices([0])

class Descriptors(object):
    """
    Launches a live interactive heatmap visualization.
    """
    def __init__(self, dcn, dataset_mean, dataset_std_dev, image_dir):
        self._cf = CorrespondenceFinder(dcn, dataset_mean, dataset_std_dev)
        self._image_dir = image_dir
        self.goal_img_path = "/Users/adivganapathi/Documents/UC Berkeley/Current Projects/cloth-rendering/cloth_images/flat_goal_rgb.png"
        #self.goal_img_path = "/home/davinci/adi/cloth-rendering/cloth_images/flat_goal_rgb.png"

    def get_new_images(self):
        """
        Gets a new pair of images
        :return:
        :rtype:
        """
        self.img1_pil = Image.open(self.goal_img_path).convert('RGB').resize((640, 480))
        #self.img1_pil = Image.open(os.path.join(self._image_dir, random.choice(os.listdir(self._image_dir)))).convert('RGB').resize((640, 480))
        self.img2_pil = Image.open(os.path.join(self._image_dir, random.choice(os.listdir(self._image_dir)))).convert('RGB').resize((640, 480))

    def compute_descriptors(self):
        """
        Computes the descriptors for image 1 and image 2 for each network
        :return:
        :rtype:
        """
        print("computing descriptors")
        self.img1 = self._cf.pil_image_to_cv2(self.img1_pil)
        self.img2 = self._cf.pil_image_to_cv2(self.img2_pil)
        self.rgb_1_tensor = self._cf.rgb_image_to_tensor(self.img1_pil)
        self.rgb_2_tensor = self._cf.rgb_image_to_tensor(self.img2_pil)
        self.img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_RGB2GRAY) / 255.0
        self.img2_gray = cv2.cvtColor(self.img2, cv2.COLOR_RGB2GRAY) / 255.0

        self._res_a = self._cf.dcn.forward_single_image_tensor(self.rgb_1_tensor).data.cpu().numpy()
        self._res_b = self._cf.dcn.forward_single_image_tensor(self.rgb_2_tensor).data.cpu().numpy()
        #self.find_best_match(None, 0, 0, None, None)

    def find_best_match(self, u, v):

        """
        For each network, find the best match in the target image to point highlighted
        with reticle in the source image. Displays the result
        :return:
        :rtype:
        """

        res_a = self._res_a
        res_b = self._res_b
        best_match_uv, best_match_diff, norm_diffs = \
            self._cf.dcn.find_best_match((u, v), res_a, res_b)
        return best_match_uv


    def run(self, u, v):
        self.get_new_images()
        self.compute_descriptors()
        best_match_uv = self.find_best_match(u, v)
        return best_match_uv

    
    def knn(self, points, error_margin, k, inputs, model=None):
        if model is None:
            model = NearestNeighbors(k, error_margin)
            model.fit(points)
        match_indices = model.kneighbors(inputs, k, return_distance=False).squeeze()
        k_matches = points[match_indices]
        return model, k_matches

if __name__ == "__main__":
    base_dir = '/nfs/diskstation/adi/models/dense_descriptor_models'
    network_dir = 'tier1_oracle_1811_consecutive_3'
    dcn = DenseCorrespondenceNetwork.from_model_folder(os.path.join(base_dir, network_dir), model_param_file=os.path.join(base_dir, network_dir, '003501.pth'))
    dcn.eval()
    #image_dir = "/nfs/diskstation/adi/datasets/descriptors_test_sets/tier1_oracle_color_test"
    image_dir = "../images"
    with open(image_dir + '/knots_info.json', 'r') as f:
        knots_info = json.load(f)
    #print(knots_info['0'])
    
    
    with open('../cfg/dataset_info.json', 'r') as f:
        dataset_stats = json.load(f)
    dataset_mean, dataset_std_dev = dataset_stats["mean"], dataset_stats["std_dev"]

    heatmap_vis = Descriptors(dcn, dataset_mean, dataset_std_dev, image_dir)
    print("starting heatmap vis")
    heatmap_vis.run()
    
    #print "ran heatmap_vis"
    cv2.destroyAllWindows()

