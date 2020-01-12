import cv2
import numpy as np
import os
import argparse

def mask(image_filename, directory):
    ''' Produces a mask of a depth image by thresholding '''
    visible_mask = cv2.imread('./%s/%s'%(directory, image_filename)).copy()
    mask = visible_mask//255
    mask_filename = image_filename.replace('visible_mask', 'mask')
    cv2.imwrite('image_masks/{}'.format(mask_filename), mask)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dir', type=str, default='image_masks')
	args = parser.parse_args()
	for filename in os.listdir('./{}'.format(args.dir)):
		try:
			print("Masking %s" % filename)
			mask(filename, args.dir)
		except:
			print("Done")
