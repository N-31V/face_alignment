#!/usr/bin/python3

import os
import argparse
import warnings
import pandas as pd
import torch
from torchvision import models
from toolkit import ModelToolkit, get_dataloaders, test_with_ced, draw_ced
from testing_dlib import test_dlib
from make_markup_csv_file import make_markup
# from net import ONet, MyNet, MyNet7
warnings.filterwarnings("ignore")


def create_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-m', '--menpo_path', type=str, required=True, help='Menpo image folder path, e.g "landmarks_task/Menpo"')
    argparser.add_argument('-w', '--w_path', type=str, required=True, help='300W image folder path, e.g "landmarks_task/300W"')
    argparser.add_argument('-d', '--description', type=str, default='ResNet50', help='describe training')
    argparser.add_argument('-e', '--epochs', type=int, default=150, help='int, default=150')
    argparser.add_argument('-n', '--num_workers', type=int, default=12, help='int, default=12')
    argparser.add_argument('-b', '--batch_size', type=int, default=16, help='int, default=16')
    return argparser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    print('making markup...')
    k = 0.4
    test_menpo_df = make_markup(os.path.join(args.menpo_path, 'test'), k)
    test_300w_df = make_markup(os.path.join(args.w_path, 'test'), k)
    train_menpo_df = make_markup(os.path.join(args.menpo_path, 'train'), k)
    train_300w_df = make_markup(os.path.join(args.w_path, 'train'), k)
    train_df = pd.concat([train_menpo_df, train_300w_df])

    train_dl, val_dl = get_dataloaders(train_df, batch_size=args.batch_size, num_workers=args.num_workers)

    print('training model...')
    # net = MyNet7()
    net = models.resnet50()
    net.fc = torch.nn.Linear(net.fc.in_features, 136)
    model = ModelToolkit(net, args.description)
    best_model_name = model.train(train_dl, val_dl, args.epochs)

    print('testing best val model...')
    model = ModelToolkit(net, args.description, checkpoint='checkpoints/{}'.format(best_model_name))

    # dlib_mse_err_m, dlib_dist_err_m = test_dlib(test_menpo_df)
    # dlib_mse_err_w, dlib_dist_err_w = test_dlib(tets_300w_df)
    net_mse_err_m, net_dist_err_m = test_with_ced(model, test_menpo_df)
    net_mse_err_w, net_dist_err_w = test_with_ced(model, test_300w_df)

    draw_ced([net_mse_err_m], ['ResNet50'], 'mse_menpo(008)')
    draw_ced([net_mse_err_w], ['ResNet50'], 'mse_300w(008)')
    # draw_ced([dlib_mse_err_m, net_mse_err_m], ['dlib', 'ResNet50'], 'mse_menpo(08)', stop=0.8)
    # draw_ced([dlib_mse_err_w, net_mse_err_w], ['dlib', 'ResNet50'], 'mse_300w(08)', stop=0.8)

    # draw_ced([dlib_dist_err_m, net_dist_err_m], ['dlib', 'ResNet50'], 'dist_menpo(008)')
    # draw_ced([dlib_dist_err_w, net_dist_err_w], ['dlib', 'ResNet50'], 'dist_300w(008)')
    # draw_ced([dlib_dist_err_m, net_dist_err_m], ['dlib', 'ResNet50'], 'dist_menpo(08)', stop=0.8)
    # draw_ced([dlib_dist_err_w, net_dist_err_w], ['dlib', 'ResNet50'], 'dist_300w(08)', stop=0.8)
