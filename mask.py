import cv2
import numpy as np
import os
import argparse

def mask(image_filename, directory, texture):
    ''' Produces a mask of a depth image by thresholding '''
    img = cv2.imread('./%s/%s'%(directory, image_filename)).copy()
    mask = img/255.
    mask_filename = image_filename.replace('visible_mask', 'mask')
    visible_mask_filename = image_filename.replace('rgb', 'visible_mask')
    cv2.imwrite('image_masks/{}'.format(mask_filename), mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='image_masks')
    parser.add_argument('--texture', action='store_true')
    args = parser.parse_args()
    for filename in os.listdir('./{}'.format(args.dir)):
    	try:
    		print("Masking %s" % filename)
    		mask(filename, args.dir, args.texture)
    	except:
            pass
