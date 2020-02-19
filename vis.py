import cv2
import numpy as np
import argparse
import os
import math
import json
import colorsys

def show_knots(idx, knots_info, save=True):
    image_filename = "{0:06d}_rgb.png".format(idx)
    img = cv2.imread('images/{}'.format(image_filename))
    pixels = knots_info[str(idx)]
    pixels = [i[0] for i in pixels]
    n = int(len(pixels)**0.5)
    vis = img.copy()
    for i, (u, v) in enumerate(pixels):
        (r, g, b) = colorsys.hsv_to_rgb(float(i)/len(pixels), 1.0, 1.0)
        R, G, B = int(255 * r), int(255 * g), int(255 * b)
        row, col = i//n, i%n
        row_symm, col_symm = n-1-row, n-1-col
        u1,v1 = pixels[row_symm*n+col]
        u2,v2 = pixels[row_symm*n+col_symm]
        u3,v3 = pixels[row*n+col_symm]
        cv2.circle(vis,(int(u), int(v)), 3, (255, 255, 255), -1)
        cv2.circle(vis,(int(u1), int(v1)), 3, (R, G, B), -1)
        cv2.circle(vis,(int(u2), int(v2)), 3, (R, G, B), -1)
        cv2.circle(vis,(int(u3), int(v3)), 3, (R, G, B), -1)
        cv2.imshow("vis", vis)
        cv2.waitKey(0)
    if save:
    	annotated_filename = "{0:06d}_annotated.png".format(idx)
    	cv2.imwrite('./annotated/{}'.format(annotated_filename), vis)


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
