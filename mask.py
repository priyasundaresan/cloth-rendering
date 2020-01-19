import cv2
import numpy as np
import os
import argparse

def mask(image_filename, directory, texture):
    ''' Produces a mask of a depth image by thresholding '''
    img = cv2.imread('./%s/%s'%(directory, image_filename)).copy()
    if not texture:
        background = (img < [35, 35, 35]).all(axis = 2)&(img > [30, 30, 30]).all(axis=2)
        img[np.where(background)] = [0, 0, 0]
        img[np.where((img > [0, 100, 100]).all(axis = 2))] = [0, 0, 0]
    else:
        background = (img < [112, 112, 112]).all(axis = 2)&(img > [30, 30, 30]).all(axis=2)
        img[np.where(background)] = [0, 0, 0]
        img[np.where((img > [176, 176, 176]).all(axis = 2))] = [0, 0, 0]
    _, binary = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    visible_mask = np.stack((binary,)*3, axis=-1) # 3-channel version of mask
    mask = visible_mask//255
    mask_filename = image_filename.replace('rgb', 'mask')
    visible_mask_filename = image_filename.replace('rgb', 'visible_mask')
    cv2.imwrite('image_masks/{}'.format(mask_filename), mask)
    cv2.imwrite('image_masks/{}'.format(visible_mask_filename), visible_mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='images')
    parser.add_argument('--texture', action='store_true')
    args = parser.parse_args()
    if not os.path.exists("./image_masks"):
    	os.makedirs('./image_masks')
    else:
    	os.system('rm -rf ./image_masks')
    	os.makedirs('./image_masks')
    for filename in os.listdir('./{}'.format(args.dir)):
    	try:
    		print("Masking %s" % filename)
    		mask(filename, args.dir, args.texture)
    	except:
            pass
