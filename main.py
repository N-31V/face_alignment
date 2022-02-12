#!/usr/bin/python3

import argparse
import warnings
import pandas as pd
import torch
from torchvision import models
from toolkit import ModelToolkit, get_dataloaders, test_with_ced, draw_ced
from testing_dlib import test_dlib
# from net import ONet, MyNet, MyNet7
warnings.filterwarnings("ignore")


def create_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-p', '--markup_path', type=str, required=True, help='csv file path')
    argparser.add_argument('-m', '--menpo_path', type=str, required=True, help='Menpo test csv file path')
    argparser.add_argument('-w', '--w_path', type=str, required=True, help='300W test csv file path')
    argparser.add_argument('-d', '--description', type=str, required=True, help='describe training')
    argparser.add_argument('-e', '--epochs', type=int, default=100, help='int, default=100')
    argparser.add_argument('-n', '--num_workers', type=int, default=12, help='int, default=12')
    argparser.add_argument('-b', '--batch_size', type=int, default=16, help='int, default=16')
    argparser.add_argument('-c', '--checkpoint', type=str, default=None, help='model checkpoint, default=None')
    return argparser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()


    train_dl, val_dl = get_dataloaders(pd.read_csv(args.markup_path, index_col=0), batch_size=args.batch_size,
                                       num_workers=args.num_workers)
    # net = MyNet9()

    print('training model...')
    net = models.resnet50()
    net.fc = torch.nn.Linear(net.fc.in_features, 136)
    model = ModelToolkit(net, args.description, checkpoint=args.checkpoint)
    best_model_name = model.train(train_dl, val_dl, args.epochs)

    print('testing best val model...')
    model = ModelToolkit(net, args.description + 'bv', checkpoint='checkpoints/{}'.format(best_model_name))

    test_menpo_df = pd.read_csv(args.menpo_path, index_col=0)
    tets_300w_df = pd.read_csv(args.w_path, index_col=0)

    dlib_mse_err_m, dlib_dist_err_m = test_dlib(test_menpo_df)
    dlib_mse_err_w, dlib_dist_err_w = test_dlib(tets_300w_df)
    net_mse_err_m, net_dist_err_m = test_with_ced(model, test_menpo_df)
    net_mse_err_w, net_dist_err_w = test_with_ced(model, tets_300w_df)

    test_with_ced(model, pd.read_csv(args.menpo_path, index_col=0))
    test_with_ced(model, pd.read_csv(args.w_path, index_col=0))

    draw_ced([dlib_mse_err_m, net_mse_err_m], ['dlib', 'ResNet50'], 'mse_menpo(008)')
    draw_ced([dlib_mse_err_w, net_mse_err_w], ['dlib', 'ResNet50'], 'mse_300w(008)')
    draw_ced([dlib_mse_err_m, net_mse_err_m], ['dlib', 'ResNet50'], 'mse_menpo(08)', stop=0.8)
    draw_ced([dlib_mse_err_w, net_mse_err_w], ['dlib', 'ResNet50'], 'mse_300w(08)', stop=0.8)

    draw_ced([dlib_dist_err_m, net_dist_err_m], ['dlib', 'ResNet50'], 'dist_menpo(008)')
    draw_ced([dlib_dist_err_w, net_dist_err_w], ['dlib', 'ResNet50'], 'dist_300w(008)')
    draw_ced([dlib_dist_err_m, net_dist_err_m], ['dlib', 'ResNet50'], 'dist_menpo(08)', stop=0.8)
    draw_ced([dlib_dist_err_w, net_dist_err_w], ['dlib', 'ResNet50'], 'dist_300w(08)', stop=0.8)
