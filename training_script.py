#coding:utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

import landmark
from landmark import Image
from feature import *
from coding import *
from gettingFile import get_all_imgfiles
from preprocessing import ImagePreprocessing

from create_cnn_model import create_alexNet_v1
from create_cnn_model import create_VGGNet_v1, create_VGGNet_v2
from setting_cnn import CNNSetting

import pdb
import json
import os
import gc
import random
import numpy as np
import cv2
import glob

from keras.utils import np_utils, generic_utils



NB_CLASS = 9
DATASETPATH = '../../data/data_ver4/data'

if __name__ == '__main__':

    preprocesser = ImagePreprocessing()
    all_imgs = preprocesser.load_image(DATASETPATH)

    train_, val_, test_ = preprocesser.split_data(all_imgs)
    # data augmentation
    train_ = preprocesser.data_augmentation(train_, flip_x=True)
    # setting the label
    y_train_ = np.asarray([each.label for each in train_])
    y_val_ = np.asarray([each.label for each in val_])
    y_test_ = np.asarray([each.label for each in test_])
    Y_train = np_utils.to_categorical(y_train_, NB_CLASS)
    Y_val = np_utils.to_categorical(y_val_, NB_CLASS)
    Y_test = np_utils.to_categorical(y_test_, NB_CLASS)
    # change the suitable format
    x_train_, y_train_ = preprocesser.img2matrix(train_)
    x_val_, y_val_ = preprocesser.img2matrix(val_)
    x_test_, y_test_ = preprocesser.img2matrix(test_)
    # preprocessing
    X_train, X_val, X_test = preprocesser.mean_subtraction(x_train_, x_val_, x_test_)
    X_train, X_val, X_test = preprocesser.normalization(X_train, X_val, X_test)
    # setting hyper-param for CNN
    model = CNNSetting(model_name=create_VGGNet_v1, n_class=NB_CLASS, fname='vggNet1_t10_data_aug')
    model.training_param(lr=0.005, layer_reg=0.0005, sgd_reg=0.05, decay=0.0005, momentum=0.9, n_epoch=200, batch_size=16, early_stop=True)
    # training CNN
    model.train(X_train,Y_train,X_val,Y_val)

    model.making_training_report()
