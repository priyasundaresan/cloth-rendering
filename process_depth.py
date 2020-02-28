import cv2
import numpy as np
import os
import numpy as np
import argparse

def process_depth(image_filename, directory):
    ''' Produces a mask of a depth image by thresholding '''
    img = cv2.imread('./%s/%s'%(directory, image_filename)).copy()
    mask = cv2.imread('./%s/%s'%('image_masks', image_filename.replace('rgb', 'visible_mask'))).copy()
    inverted = 255 - img

    # Uncomment for darkening
    #darkened = np.double(inverted) - np.random.randint(40, 90)
    #darkened[np.where((darkened < 0))] = 0
    #darkened = np.uint8(darkened)
    #inverted = darkened

    cv2.imshow("img", inverted)
    cv2.waitKey(0)
    #mask_filename = image_filename.replace('visible_mask', 'mask')
    #cv2.imwrite('image_masks/{}'.format(mask_filename), mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='images_depth')
    args = parser.parse_args()
    for filename in os.listdir('./{}'.format(args.dir)):
    	print("Masking %s" % filename)
    	process_depth(filename, args.dir)
