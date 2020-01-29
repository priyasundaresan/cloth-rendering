from descriptors.dense_correspondence_network import DenseCorrespondenceNetwork
import pprint
import json
import sys
import os
import cv2
import numpy as np
import copy
from PIL import Image
from torchvision import transforms
#from sklearn.neighbors import NearestNeighbors

#from image_utils import geometric_median, sample_nearest_points_on_mask, farthest_pixel_correspondence
#from image_utils import * 
#from pixel_selector import PixelSelector

class CorrespondenceFinder:
    def __init__(self, dcn, dataset_mean, dataset_std_dev, flip_horizontal=False):
        self.dcn = dcn
        self.dataset_mean = dataset_mean
        self.dataset_std_dev = dataset_std_dev
        self.flip_horizontal=flip_horizontal
        self.img1_flipped = False
        self.img2_flipped = False

    def get_rgb_image(self, rgb_filename):
        """
        :param depth_filename: string of full path to depth image
        :return: PIL.Image.Image, in particular an 'RGB' PIL image
        """
        return Image.open(rgb_filename).convert('RGB').resize((640, 480))

    def get_grayscale_image(self, grayscale_filename):
        return Image.open(grayscale_filename).resize((640, 480))

    def pil_image_to_cv2(self, pil_image):
        return np.array(pil_image)[:, :, ::-1].copy() # open and convert between BGR and RGB
    
    def rgb_image_to_tensor(self, img):
        norm_transform = transforms.Normalize(self.dataset_mean, self.dataset_std_dev) 
        return transforms.Compose([transforms.ToTensor(), norm_transform])(img)

    def load_image_pair(self, img1_filename, img2_filename):
        self.img1_pil = self.get_rgb_image(img1_filename)
        self.img2_pil = self.get_rgb_image(img2_filename)
        #print "loaded images successfully"

    def flip_images(self):
        circle1 = locate_circle_center_hough(self.img1)
        circle2 = locate_circle_center_hough(self.img2)
        if circle1 is not None:
            u1, v1 = circle1
            if u1 > 320:
                self.img1_pil = self.img1_pil.transpose(Image.FLIP_LEFT_RIGHT)
                #print "flipped 1st"
                self.img1_flipped = True
        if self.img1_flipped:
            #print "flipped 2nd"
            self.img2_pil = self.img2_pil.transpose(Image.FLIP_LEFT_RIGHT)
            self.img2_flipped = True
        #if circle2 is not None:
        #    u2, v2 = circle2
        #    if u2 > 320:
        #        #print "flipped 2nd"
        #        self.img2_pil = self.img2_pil.transpose(Image.FLIP_LEFT_RIGHT)
        #        self.img2_flipped = True

    def compute_descriptors(self):
        self.img1 = self.pil_image_to_cv2(self.img1_pil)
        self.img2 = self.pil_image_to_cv2(self.img2_pil)
        if self.flip_horizontal:
            self.flip_images()
        self.rgb_1_tensor = self.rgb_image_to_tensor(self.img1_pil)
        self.rgb_2_tensor = self.rgb_image_to_tensor(self.img2_pil)
        self.img1_descriptor = self.dcn.forward_single_image_tensor(self.rgb_1_tensor).data.cpu().numpy()
        self.img2_descriptor = self.dcn.forward_single_image_tensor(self.rgb_2_tensor).data.cpu().numpy()

    def find_k_best_matches(self, pixels, k, mode="median", annotate=True):
        # Finds k best matches in descriptor space (either by median or mean filtering)
        max_range = float(len(pixels))
        pixel_matches = []
        if self.flip_horizontal:
            if self.img1_flipped:
                pixels = flip_pixels_horizontal(pixels)
        model = None
        # best_matches, norm_diffs, model = self.dcn.find_best_match_for_descriptors_KNN(np.array(pixels), self.img1_descriptor, self.img2_descriptor, k)
        for i, (u, v) in enumerate(pixels):
            if mode == "median":
                best_matches, norm_diffs, norm_diffs_all = self.dcn.find_k_best_matches((u, v), self.img1_descriptor, self.img2_descriptor, k)
                best_match = np.round(np.median(best_matches, axis=0))
            elif mode == "geometric_median_cloud":
                cloud = sample_nearest_points_on_mask((u, v), self.img1, 10)
                best_matches, norm_diffs, model = self.dcn.find_best_match_for_descriptors_KNN(cloud, self.img1_descriptor, self.img2_descriptor, 1, model)
                best_match = geometric_median(best_matches.squeeze())
            else:
                best_matches, norm_diffs, norm_diffs_all = self.dcn.find_k_best_matches((u, v), self.img1_descriptor, self.img2_descriptor, k)
                best_match = np.round(np.mean(best_matches, axis=0))
            match = [int(best_match[0]), int(best_match[1])]
            pixel_matches.append(match)

        if self.flip_horizontal:
            if self.img1_flipped:
                pixels = flip_pixels_horizontal(pixels)
            if self.img2_flipped:
                pixel_matches = flip_pixels_horizontal(pixel_matches)
        for i, (u, v) in enumerate(pixels):
            match = pixel_matches[i]
            if annotate:
                self.annotate_correspondence(u, v, match[0], match[1])

        #idxs = prune_close_pixel_indices(pixel_matches)
        #idxs = []
        #pixels = [pixels[i] for i in range(len(pixels)) if not i in idxs]
        #pixel_matches = [pixel_matches[i] for i in range(len(pixel_matches)) if not i in idxs]
        self.img1_flipped = False
        self.img2_flipped = False
        return pixel_matches, pixels

    def find_best_match_pixel(self, u, v):
        best_match_uv, best_match_diff, norm_diffs = \
                self.dcn.find_best_match((u, v), self.img1_descriptor, self.img2_descriptor)
        return (best_match_uv, best_match_diff, norm_diffs)

    def find_best_matches_raw(self, pixels):
        max_range = float(len(pixels))
        for i, (u, v) in enumerate(pixels):
            best_match, norm_diff, norm_diffs = self.find_best_match_pixel(u, v)
            self.annotate_correspondence(u, v, int(best_match[0]), int(best_match[1]))

    def annotate_correspondence(self, u1, v1, u2, v2, line=False, flip=False):
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        src = self.img1
        dest = self.img2
        if flip:
            src = self.img2
            dest = self.img1
        cv2.circle(src, (u1, v1), 4, color, -1)
        cv2.circle(dest, (u2, v2), 4, color, -1)
        if line:
            cv2.line(src, (u1, v1), (u2, v2), (255, 255, 255), 4)


    def create_circular_mask(self, center, h=480, w=640, radius=10):
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        mask = dist_from_center <= radius
        return mask

    def show_side_by_side(self, flip=False):
        if flip:
            vis = np.concatenate((self.img2, self.img1), axis=1)
        else:
            vis = np.concatenate((self.img1, self.img2), axis=1)
        cv2.imshow("correspondence", vis)
        k = cv2.waitKey(0)
        if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
        return vis


if __name__ == '__main__':
    base_dir = '/nfs/diskstation/priya/rope_networks'   
    #network_dir = 'rope_noisy_1400_depth_norm_3'
    network_dir = 'rope_557_task_loops_16'
    dcn = DenseCorrespondenceNetwork.from_model_folder(os.path.join(base_dir, network_dir), model_param_file=os.path.join(base_dir, network_dir, '003501.pth'))
    dcn.eval()
    with open('../cfg/dataset_info.json', 'r') as f:
        dataset_stats = json.load(f)
    dataset_mean, dataset_std_dev = dataset_stats["mean"], dataset_stats["std_dev"]

    cf = CorrespondenceFinder(dcn, dataset_mean, dataset_std_dev, flip_horizontal=True)
    ps = PixelSelector()
    pixels = None
    f1 = '../loop_reference/phoxi/segdepth_0.png'
    for i in range(1, 15, 2):
        f2 = '../images/phoxi/segdepth_%d.png' % i
        #print f1, f2
        cf.load_image_pair(f1, f2)
        cf.compute_descriptors()
        if pixels is None:
            pixels = ps.run(cf.img1)
        #pixels = sample_sparse_points(cf.pil_image_to_cv2(cf.img1), k=100, dist=50)
        best_matches, _ = cf.find_k_best_matches(pixels, 100, mode="median", annotate=True)
        end_match = best_matches[0]
        loop_match = best_matches[1]
        cf.annotate_correspondence(loop_match[0], loop_match[1], end_match[0], loop_match[1], line=True, flip=True)
        #farthest_correspondence = farthest_pixel_correspondence(pixels, best_matches)
        #src_px, target_px = farthest_correspondence
        vis = cf.show_side_by_side(flip=True)
