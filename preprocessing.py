#coding:utf-8
import landmark
from landmark import Image
from gettingFile import get_all_imgfiles

from sklearn.cross_validation import train_test_split
from keras.utils import np_utils, generic_utils

import numpy as np
import cv2
import copy

class ImagePreprocessing(object):

    def load_image(self,path):
        all_paths = get_all_imgfiles(path)
        all_objImage = [Image(p) for p in all_paths]
        return all_objImage

    def split_data(self,data):
        """ split data to training set and test set
        data: list of objImage
        return : list of objImage spliting to train, val, test 3 parts
        """
        label_ = [each.label for each in data]
        X_train_, X_test, Y_train_, Y_test = train_test_split(data,label_,random_state=0)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train_,Y_train_,train_size=0.7, random_state=0)
        return X_train, X_val, X_test

    def img2matrix(self,img_list):
        """
        img_list: list of objImage
        """
        n_data = len(img_list)
        data, label = [], []
        for i,each in enumerate(img_list):
            #img = cv2.imread(each.path)
            img = each.img
            img = cv2.resize(img,(224,224)) # should be improved
            img_rgb = img.swapaxes(0,2).swapaxes(1,2) #(3,height,width)
            arr = np.asarray(img_rgb,dtype="float32")
            data.append(arr)
            label.append(each.label)
        data = np.asarray(data)
        label = np.asarray(label)
        return data,label

    def mean_subtraction(self,X1, X2, X3):
        """ its also called zero-centered
        X1: array of m*n, training data
        X2: array, valid data
        X3: array, test data
        """
        mean = np.mean(X1, axis=0)
        X1 -= mean
        X2 -= mean
        X3 -= mean
        return X1, X2, X3

    def normalization(self,X1, X2, X3):
        """
        X1: array, training data
        X2: array, valid data
        X3: array, test data
        """
        #scale = np.max(data)
        #data /= scale
        scale = np.std(X1, axis=0)
        X1 /= scale
        X2 /= scale
        X3 /= scale
        return X1, X2, X3

    def data_augmentation(self, data, flip_x=None):
        """ now only able to flip image around x-axis
        data: a list of objImage
        """
        augment = []

        if flip_x:
            for each in data:
                aug_flip = copy.copy(each)
                aug_flip.flip_x()
                augment.append(aug_flip)
            print "Data augmentation: flipping each image around the x-axis"
        final = augment + data
        return final
