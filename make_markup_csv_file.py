#!/usr/bin/python3

import os
import glob
import argparse
import dlib
import numpy as np
import pandas as pd
from tqdm import tqdm


def create_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-p', '--image_path', type=str, required=True, help='faces folder path')
    return argparser


def make_markup(image_path, k):
    detector = dlib.get_frontal_face_detector()
    markups = {}

    for f in tqdm(glob.glob(os.path.join(image_path, '*.jpg'))):
        img = dlib.load_rgb_image(f)
        pts_file = f.split('.')[0] + '.pts'
        with open(pts_file) as file:
            lines = file.readlines()
        if lines[1] == 'n_points: 68\n':
            lines = lines[3:-1]
            points = np.fromstring(''.join(lines), sep=' ') - 1 # matlab convention of 1 being the first index -> 0
            xs = points[0::2]
            ys = points[1::2]

            boxes = detector(img, 1)
            for box in boxes:
                height = box.bottom() - box.top()
                width = box.right() - box.left()
                # ymin = int(max(0, box.top() - k * height))
                # ymax = int(min(img.shape[0], box.bottom() + k * height))
                # xmin = int(max(0, box.left() - k*width))
                # xmax = int(min(img.shape[1], box.right() + k*width))
                if height > width:
                    ymin = int(max(0, box.top() - k * height))
                    ymax = int(min(img.shape[0], box.bottom() + k * height))
                    dx = (ymax - ymin - width) / 2
                    xmin = int(max(0, box.left() - dx))
                    xmax = int(min(img.shape[1], box.right() + dx))
                else:
                    xmin = int(max(0, box.left() - k*width))
                    xmax = int(min(img.shape[1], box.right() + k*width))
                    dy = (xmax - xmin - height) / 2
                    ymin = int(max(0, box.top() - dy))
                    ymax = int(min(img.shape[0], box.bottom() + dy))
                if np.all(xs > xmin) and np.all(xs < xmax) and np.all(ys > ymin) and np.all(ys < ymax):
                    markups[f] = [box.right(), box.left(), box.top(), box.bottom(), xmin, xmax, ymin, ymax] + \
                                                xs.tolist() + ys.tolist()
    columns = ['right', 'left', 'top', 'bottom', 'xmin', 'xmax', 'ymin', 'ymax']
    columns += ['x{}'.format(i+1) for i in range(68)] + ['y{}'.format(i+1) for i in range(68)]
    data = pd.DataFrame.from_dict(markups, orient='index', columns=columns)
    return data


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    k = 0.4
    data = make_markup(args.image_path, k)
    print(k, len(data))
    data.to_csv(args.image_path + '.csv')








