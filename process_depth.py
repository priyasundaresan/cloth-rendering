import cv2
import numpy as np
import os
import argparse

def invert(image_filename, directory):
    ''' Produces a mask of a depth image by thresholding '''
    img = cv2.imread('./%s/%s'%(directory, image_filename)).copy()
    inverted = 255 - img
    cv2.imwrite('{}/{}'.format(directory, image_filename), inverted)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='images_depth')
    parser.add_argument('--texture', action='store_true')
    args = parser.parse_args()
    for filename in os.listdir('./{}'.format(args.dir)):
    	print("Masking %s" % filename)
    	invert(filename, args.dir)
    	#try:
    	#	print("Masking %s" % filename)
    	#	mask(filename, args.dir, args.texture)
    	#except:
        #    pass
