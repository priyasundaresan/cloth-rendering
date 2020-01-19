import cv2
import numpy as np
import argparse
import os
import math
import json
import colorsys

def show_knots(idx, knots_info, save=True):
    image_filename = "{0:06d}_rgb.png".format(idx)
    #image_filename = "{0:06d}.jpg".format(idx)
    #img = cv2.imread('images/{}'.format(image_filename)) * 0
    img = cv2.imread('images/{}'.format(image_filename))
    pixels = knots_info[str(idx)]
    for i in range(len(pixels)):
        for (u, v) in pixels[i]:
            (r, g, b) = colorsys.hsv_to_rgb(float(i)/len(pixels), 1.0, 1.0)
            R, G, B = int(255 * r), int(255 * g), int(255 * b)
            cv2.circle(img,(int(u), int(v)), 1, (R, G, B), -1)
    if save:
    	annotated_filename = "{0:06d}_annotated.png".format(idx)
    	cv2.imwrite('./annotated/{}'.format(annotated_filename), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=len(os.listdir('./images')) - 1)
    args = parser.parse_args()
    if not os.path.exists("./annotated"):
        os.makedirs('./annotated')
    else:
        os.system("rm -rf ./annotated")
        os.makedirs("./annotated")
    print("parsed")
    with open("images/knots_info.json", "r") as stream:
    	knots_info = json.load(stream)
    print("loaded knots info")
    for i in range(args.num):
        print(i)
        show_knots(i, knots_info)
