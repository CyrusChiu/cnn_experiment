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

from create_cnn_model import create_VGGNet_v1, create_alexNet_v1

import pdb
import json
import os
import gc
import random
import numpy as np
import cv2
import glob

from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD



#def zca_whitening(X):
#    """ Remember that the data matrix should be zero-center
#    X: m*n array of m data, n feature
#    """
#    X_ = X.reshape(X.shape[0],X.shape[1]*X.shape[2]*X.shape[3])
#    pdb.set_trace()
#    cov = np.dot(X_.T, X_) / X_.shape[0]
#    U,S,V = np.linalg.svd(cov)
#    Xrot = np.dot(X_, U)
#    XPCAwhite = Xrot / np.sqrt(S + 1e-5)
#    XZCAwhite = np.dot(U,XPCAwhite)
#    return XZCAwhite




class CNNSetting(object):
    def __init__(self, model_name, n_class, fname, model=None, n_epoch=None, batch_size=None, lr=None, layer_reg=None, sgd_reg=None, decay=None, momentum=None, weights=None, early_stop=None, early_stop_msg=None, records=None):
        self.model_name = model_name
        self.n_class = n_class
        self.fname = fname
        self.model = model
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.layer_reg = layer_reg
        self.sgd_reg = sgd_reg
        self.decay = decay
        self.momentum = momentum
        self.weights = weights
        self.early_stop = early_stop
        self.early_stop_msg = early_stop_msg
        self.records = records

    def training_param(self, lr, layer_reg,  sgd_reg, decay, momentum, n_epoch, batch_size, early_stop):
        self.lr = lr
        self.layer_reg = layer_reg
        self.sgd_reg = sgd_reg
        self.decay = decay
        self.momentum = momentum
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.early_stop = early_stop
        model = self.model_name(self.n_class, self.layer_reg)
        sgd = SGD(l2=sgd_reg, lr=lr, decay=decay, momentum=momentum, nesterov=True)

        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        self.model = model

    def train(self,x_train,y_train,x_val,y_val):
        best_accuracy = 0.0
        model_weights = []
        model = self.model
        batch_size = self.batch_size
        t_cost, v_cost, t_acc, v_acc = [], [], [], []

        for e in range(self.n_epoch):
            x_train_, y_train_ = self._shuffling_data_(x_train, y_train)
            n_batch = len(y_train_) / batch_size
            train_loss, train_accuracy = 0.0, 0.0

            for i in range(n_batch):
                loss,accuracy = model.train_on_batch(x_train_[i*batch_size:(i+1)*batch_size],
                                                     y_train_[i*batch_size:(i+1)*batch_size],
                                                     accuracy=True)
                train_loss += loss
                train_accuracy += accuracy

            train_loss = train_loss / n_batch
            train_accuracy = train_accuracy / n_batch

            val_loss,val_accuracy = model.evaluate(x_val, y_val, batch_size=1, show_accuracy=True, verbose=0)
            t_cost.append(round(train_loss,5))
            v_cost.append(round(val_loss,5))
            t_acc.append(round(train_accuracy,5))
            v_acc.append(round(val_accuracy,5))

            line = "epoch %s / %s:  cost %s  train %s  val %s" %(str(e+1), str(self.n_epoch), str(round(train_loss,5)), str(round(train_accuracy,5)), str(round(val_accuracy,5)))
            print line

            if best_accuracy < val_accuracy:
                best_accuracy = val_accuracy
                fname = self.fname +'_lr'+str(self.lr)+'_layerReg'+str(self.layer_reg) +'_sgdReg'+ str(self.sgd_reg)  +'_e'+str(e+1)+'_acc'+str(round(best_accuracy,3))+'.cnn'
                model_weights.append(fname)
                model.save_weights(fname)

            # early stopping
            # if the latest 3 records of training acc are 1.0
            if self.early_stop == True:
                if t_acc[::-1][:5] == [1.0, 1.0, 1.0, 1.0, 1.0]:
                    # and if the latest val acc is less then previous one
                    if v_acc[::-1][0] <= v_acc[::-1][1]:
                        msg = 'early stopping at epoch %s' %(str(e+1))
                        self.early_stop_msg = msg
                        print msg
                        break

        top_5 = model_weights[::-1][:5]
        to_del = list(set(model_weights) - set(top_5))
        for w in to_del:
            os.remove(w)

        records = {
                  'model name' : str(self.fname),
                  'learning rate' : str(self.lr),
                  'sgd reg' : str(self.sgd_reg),
                  'layer reg' : str(self.layer_reg),
                  'batch size' : str(self.batch_size),
                  'training loss' : t_cost,
                  'training accuracy' : t_acc,
                  'validation loss' : v_cost,
                  'validation accuracy' : v_acc }

        if self.early_stop_msg:
            records.update({'early stop':self.early_stop_msg})
        else:
            records.update({'early stop':'None'})

        record_fname = self.fname +'_lr'+str(self.lr)+'.json'
        with open(record_fname,'w') as jfile:
            json.dump(records, jfile)


        self.weights = top_5
        self.records = records

    def making_training_report(self, figure=True):

        r1 = "model name: " + self.fname
        r2 =  "learning rate: " + str(self.lr)
        r3 =  "gradient descend: mini-batch SGD"
        r4 =  "batch size: " + str(self.batch_size)
        r5 =  "training epoch: " + str(self.n_epoch)
        r6 =  "regularization of conv layer: " + str(self.layer_reg)
        r7 =  "regularization of SGD: " + str(self.sgd_reg)

        print r1
        print r2
        print r3
        print r4
        print r5
        print r6
        print r7

        if self.early_stop_msg:
            print self.early_stop_msg

        if figure:
            records = self.records
            plt.figure()
            plt.plot(records['training loss'],'-', label='training')
            plt.plot(records['validation loss'],'-', label='validation')
            #plt.legend(loc='upper right')
            plt.legend(loc='best')
            plt.ylabel('Cost')
            plt.xlabel('Epochs')
            fname = 'cost_' + self.fname +'_lr'+str(self.lr)+'.png'
            plt.savefig(fname)

            plt.figure()
            plt.plot(records['training accuracy'],'-', label='training')
            plt.plot(records['validation accuracy'],'-', label='validation')
            #plt.legend(loc='upper left')
            plt.legend(loc='best')
            plt.ylabel('Accuracy')
            plt.xlabel('Epochs')
            fname = 'acc_' + self.fname +'_lr'+str(self.lr)+'.png'
            plt.savefig(fname)


    def _init_model_for_predict_(self):
        lr = self.lr
        sgd_reg = self.sgd_reg
        layer_reg = self.layer_reg
        decay = self.decay
        momentum = self.momentum
        model = self.model_name(self.n_class, layer_reg)
        sgd = SGD(l2=sgd_reg, lr=lr, decay=decay, momentum=momentum, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        return model


    def predict(self,x_test,y_test):
        for weight in self.weights:
            print 'prediction on ' + str(weight)
            model = self._init_model_for_predict_()
            model.load_weights(weight)
            y_pred = model.predict_classes(x_test, batch_size=1, verbose=0)
            print classification_report(y_test, y_pred)
            print '\n'
            del model


    def _shuffling_data_(self,x,y):
        idx = [i for i in range(len(y))]
        random.shuffle(idx)
        x = x[idx] # data
        y = y[idx] # label
        return x, y
