#!/usr/bin/python3

import dlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from toolkit import count_mse_err, count_dist_err, draw_ced


def create_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-p', '--markup_path', type=str, required=True, help='csv file path')
    return argparser


def test_dlib(df):
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)

    dist_err = []
    mse_err = []
    for f in tqdm(df.index):
        img = dlib.load_rgb_image(f)
        box = dlib.rectangle(df.loc[f, 'left'], df.loc[f, 'top'], df.loc[f, 'right'], df.loc[f, 'bottom'])
        points = predictor(img, box)
        x_pred = []
        y_pred = []
        for i in range(68):
            point = points.part(i)
            x_pred.append(point.x)
            y_pred.append(point.y)
        dist_err.append(count_dist_err(
            np.array(x_pred),
            np.array(y_pred),
            df.loc[f, 'x1':'x68'].to_numpy(),
            df.loc[f, 'y1':'y68'].to_numpy()
        ))
        mse_err.append(count_mse_err(
            np.array(x_pred + y_pred),
            df.loc[f, 'x1':'y68'].to_numpy(),
            np.sqrt((df.loc[f, 'bottom'] - df.loc[f, 'top']) * (df.loc[f, 'right'] - df.loc[f, 'left']))
        ))
    dist_err = np.sort(dist_err)
    mse_err = np.sort(mse_err)
    path = df.index[0].split('/')
    # draw_ced(mse_err, 'dlib_mse_square_{}_{}'.format(path[1], path[2]))
    # draw_ced(dist_err, 'dlib_dist_square_{}_{}'.format(path[1], path[2]))
    # draw_ced(mse_err, 'dlib_mse_square_8_{}_{}'.format(path[1], path[2]), stop=0.8)
    # draw_ced(dist_err, 'dlib_dist_square_8_{}_{}'.format(path[1], path[2]), stop=0.8)
    return mse_err, dist_err


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    test_dlib(pd.read_csv(args.markup_path, index_col=0))
