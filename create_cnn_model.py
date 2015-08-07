#coding:utf-8
from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.regularizers import l2, activity_l2, WeightRegularizer


def create_VGGNet_v3(nb_class):
    """ This is the original VGGNet
    My computer can not handle this right now
    got memory error problem
    nb_class = # of class
    base on
    VGGNet / OxfordNet (runner-up winner of ILSVRC 2014)
    [Simonyan and Zisserman]
    """
    model = Sequential()
    # weights layer 1
    # (64,3,3,3) -> #of receptive field ,RGB ,rf size 3*3
    # subsample=(1,1) -> stride=1
    # default = subsample=(1,1)
    model.add(Convolution2D(64, 3, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 2
    # (64,64,3,3) -> 64 rf ,input=64 ,rf size 3*3
    model.add(Convolution2D(64, 64, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # pooling layer 1
    # image size from 224 to 112
    model.add(MaxPooling2D(poolsize=(2, 2)))

    # weights layer 3
    # (128,64,3,3) -> 128 rf , input=64 ,rf size 3*3
    model.add(Convolution2D(128, 64, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 4
    # (128,128,3,3) -> 128 rf , input=128 ,rf size 3*3
    model.add(Convolution2D(128, 128, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # pooling layer 2
    # image size from 112 to 56
    model.add(MaxPooling2D(poolsize=(2, 2)))

    # weights layer 5
    # (256,128,3,3) -> 256 rf , input=128 ,rf size 3*3
    model.add(Convolution2D(256, 128, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 6
    # (256,256,3,3) -> 256 rf , input=256 ,rf size 3*3
    model.add(Convolution2D(256, 256, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 7
    # (256,256,3,3) -> 256 rf , input=256 ,rf size 3*3
    model.add(Convolution2D(256, 256, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # pooling layer 3
    # image size from 56 to 28
    model.add(MaxPooling2D(poolsize=(2, 2)))

    # weights layer 8
    # (512,256,3,3) -> 512 rf , input=256 ,rf size 3*3
    model.add(Convolution2D(512, 256, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 9
    # (512,512,3,3) -> 512 rf , input=512 ,rf size 3*3
    model.add(Convolution2D(512, 512, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 10
    # (512,512,3,3) -> 512 rf , input=512 ,rf size 3*3
    model.add(Convolution2D(512, 512, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # pooling layer 4
    # image size from 28 to 14
    model.add(MaxPooling2D(poolsize=(2, 2)))

    # weights layer 11
    # (512,512,3,3) -> 512 rf , input=512 ,rf size 3*3
    model.add(Convolution2D(512, 512, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 12
    # (512,512,3,3) -> 512 rf , input=512 ,rf size 3*3
    model.add(Convolution2D(512, 512, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 13
    # (512,512,3,3) -> 512 rf , input=512 ,rf size 3*3
    model.add(Convolution2D(512, 512, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # pooling layer 5
    # image size from 14 to 7
    model.add(MaxPooling2D(poolsize=(2, 2)))

    # fc layer 1
    model.add(Flatten())
    model.add(Dense(512, 1024, init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # fc layer 2
    model.add(Dense(1024, 1024, init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # fc layer 3
    model.add(Dense(1024, nb_class, init='normal'))
    # softmax layer
    model.add(Activation('softmax'))

    return model

def create_VGGNet_v1(nb_class, layer_reg):
    """ This model is half of original VGGNet
    with reg at conv layers
    nb_class = # of class
    layer_reg : regularizers of conv layers, float, e.g. 0.01
    base on
    VGGNet / OxfordNet (runner-up winner of ILSVRC 2014)
    [Simonyan and Zisserman]
    """
    model = Sequential()
    # weights layer 1
    # (64,3,3,3) -> #of receptive field ,RGB ,rf size 3*3
    # subsample=(1,1) -> stride=1
    # default = subsample=(1,1)
    #model.add(Convolution2D(64, 3, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(32, 3, 3, 3, W_regularizer=l2(layer_reg), init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 2
    # (64,64,3,3) -> 64 rf ,input=64 ,rf size 3*3
    #model.add(Convolution2D(64, 64, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(32, 32, 3, 3, W_regularizer=l2(layer_reg), init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # pooling layer 1
    # image size from 224 to 112
    model.add(MaxPooling2D(poolsize=(2, 2)))

    # weights layer 3
    # (128,64,3,3) -> 128 rf , input=64 ,rf size 3*3
    #model.add(Convolution2D(128, 64, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(64, 32, 3, 3, W_regularizer=l2(layer_reg), init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 4
    # (128,128,3,3) -> 128 rf , input=128 ,rf size 3*3
    #model.add(Convolution2D(128, 128, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(64, 64, 3, 3, W_regularizer=l2(layer_reg), init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # pooling layer 2
    # image size from 112 to 56
    model.add(MaxPooling2D(poolsize=(2, 2)))

    # weights layer 5
    # (256,128,3,3) -> 256 rf , input=128 ,rf size 3*3
    #model.add(Convolution2D(256, 128, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(128, 64, 3, 3, W_regularizer=l2(layer_reg), init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 6
    # (256,256,3,3) -> 256 rf , input=256 ,rf size 3*3
    #model.add(Convolution2D(256, 256, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(128, 128, 3, 3, W_regularizer=l2(layer_reg), init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 7
    # (256,256,3,3) -> 256 rf , input=256 ,rf size 3*3
    #model.add(Convolution2D(256, 256, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(128, 128, 3, 3, W_regularizer=l2(layer_reg), init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # pooling layer 3
    # image size from 56 to 28
    model.add(MaxPooling2D(poolsize=(2, 2)))

    # weights layer 8
    # (512,256,3,3) -> 512 rf , input=256 ,rf size 3*3
    #model.add(Convolution2D(512, 256, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(256, 128, 3, 3, W_regularizer=l2(layer_reg), init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 9
    # (512,512,3,3) -> 512 rf , input=512 ,rf size 3*3
    #model.add(Convolution2D(512, 512, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(256, 256, 3, 3, W_regularizer=l2(layer_reg), init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 10
    # (512,512,3,3) -> 512 rf , input=512 ,rf size 3*3
    #model.add(Convolution2D(512, 512, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(256, 256, 3, 3, W_regularizer=l2(layer_reg), init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # pooling layer 4
    # image size from 28 to 14
    model.add(MaxPooling2D(poolsize=(2, 2)))

    # weights layer 11
    # (512,512,3,3) -> 512 rf , input=512 ,rf size 3*3
    #model.add(Convolution2D(512, 512, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(256, 256, 3, 3, W_regularizer=l2(layer_reg), init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 12
    # (512,512,3,3) -> 512 rf , input=512 ,rf size 3*3
    #model.add(Convolution2D(512, 512, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(256, 256, 3, 3, W_regularizer=l2(layer_reg), init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 13
    # (512,512,3,3) -> 512 rf , input=512 ,rf size 3*3
    #model.add(Convolution2D(512, 512, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(256, 256, 3, 3, W_regularizer=l2(layer_reg), init='he_normal', border_mode='valid'))
    model.add(Activation('relu'))

    # pooling layer 5
    # image size from 14 to 7
    model.add(MaxPooling2D(poolsize=(2, 2)))

    # fc layer 1
    model.add(Flatten())
    model.add(Dense(256, 512, init='normal'))
    model.add(Activation('relu'))
    # if you want to fit tiny dataset, close dropout
    model.add(Dropout(0.5))

    # fc layer 2
    model.add(Dense(512, 512, init='normal'))
    model.add(Activation('relu'))
    # if you want to fit tiny dataset, close dropout
    model.add(Dropout(0.5))

    # fc layer 3
    model.add(Dense(512, nb_class, init='normal'))
    # softmax layer
    model.add(Activation('softmax'))

    return model


def create_VGGNet_v2(nb_class):
    """ difference between v1 is half the receptive field numbers
    (1/4 of original)
    nb_class = # of class
    base on
    VGGNet / OxfordNet (runner-up winner of ILSVRC 2014)
    [Simonyan and Zisserman]
    """
    model = Sequential()
    # weights layer 1
    # (64,3,3,3) -> #of receptive field ,RGB ,rf size 3*3
    # subsample=(1,1) -> stride=1
    # default = subsample=(1,1)
    #model.add(Convolution2D(64, 3, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(16, 3, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 2
    # (64,64,3,3) -> 64 rf ,input=64 ,rf size 3*3
    #model.add(Convolution2D(64, 64, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(16, 16, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))

    # pooling layer 1
    # image size from 224 to 112
    model.add(MaxPooling2D(poolsize=(2, 2)))

    # weights layer 3
    # (128,64,3,3) -> 128 rf , input=64 ,rf size 3*3
    #model.add(Convolution2D(128, 64, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(32, 16, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 4
    # (128,128,3,3) -> 128 rf , input=128 ,rf size 3*3
    #model.add(Convolution2D(128, 128, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(32, 32, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))

    # pooling layer 2
    # image size from 112 to 56
    model.add(MaxPooling2D(poolsize=(2, 2)))

    # weights layer 5
    # (256,128,3,3) -> 256 rf , input=128 ,rf size 3*3
    #model.add(Convolution2D(256, 128, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(64, 32, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 6
    # (256,256,3,3) -> 256 rf , input=256 ,rf size 3*3
    #model.add(Convolution2D(256, 256, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(64, 64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 7
    # (256,256,3,3) -> 256 rf , input=256 ,rf size 3*3
    #model.add(Convolution2D(256, 256, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(64, 64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))

    # pooling layer 3
    # image size from 56 to 28
    model.add(MaxPooling2D(poolsize=(2, 2)))

    # weights layer 8
    # (512,256,3,3) -> 512 rf , input=256 ,rf size 3*3
    #model.add(Convolution2D(512, 256, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(128, 64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 9
    # (512,512,3,3) -> 512 rf , input=512 ,rf size 3*3
    #model.add(Convolution2D(512, 512, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(128, 128, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 10
    # (512,512,3,3) -> 512 rf , input=512 ,rf size 3*3
    #model.add(Convolution2D(512, 512, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(128, 128, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))

    # pooling layer 4
    # image size from 28 to 14
    model.add(MaxPooling2D(poolsize=(2, 2)))

    # weights layer 11
    # (512,512,3,3) -> 512 rf , input=512 ,rf size 3*3
    #model.add(Convolution2D(512, 512, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(128, 128, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 12
    # (512,512,3,3) -> 512 rf , input=512 ,rf size 3*3
    #model.add(Convolution2D(512, 512, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(128, 128, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))

    # weights layer 13
    # (512,512,3,3) -> 512 rf , input=512 ,rf size 3*3
    #model.add(Convolution2D(512, 512, 3, 3, init='he_normal', border_mode='valid'))
    model.add(Convolution2D(128, 128, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))

    # pooling layer 5
    # image size from 14 to 7
    model.add(MaxPooling2D(poolsize=(2, 2)))

    # fc layer 1
    model.add(Flatten())
    model.add(Dense(128, 256, init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # fc layer 2
    model.add(Dense(256, 256, init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # fc layer 3
    model.add(Dense(256, nb_class, init='normal'))
    # softmax layer
    model.add(Activation('softmax'))

    return model

def create_alexNet_v1(nb_class):
    """
    這個是src5.2.2在用的舊model
    設計的原則是alexNet的砍半
    """

    model = Sequential()
    # AlexNet用2顆 gpu，所以kernel數目都是我的兩倍，我下面是48，他是96
    # AlexNet c1層，48個11x11的kernel對224x224的輸入做conv，stride(滑動步長)為4，
    # subsample=(4,4)就是滑動步長=4的意思，預設值是(1,1)
    model.add(Convolution2D(48, 3, 11, 11, border_mode='valid',subsample=(4,4)))
    model.add(Activation('relu'))
    # layer S2
    model.add(MaxPooling2D(poolsize=(2, 2)))

    # layer C3
    # 128個 5x5x48的kernel
    model.add(Convolution2D(128,48, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    # layer S4
    model.add(MaxPooling2D(poolsize=(2, 2)))

    # layer C5
    model.add(Convolution2D(192, 128, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(poolsize=(2, 2)))

    # layer C6
    model.add(Convolution2D(192, 192, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))

    # layer C7
    model.add(Convolution2D(128, 192, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    # layer S8
    model.add(MaxPooling2D(poolsize=(2, 2)))

    # 9 全連接層
    model.add(Flatten())
    model.add(Dense(128*2*2, 512, init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # 10 全連接層
    #model.add(Flatten())
    model.add(Dense(512, 512, init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # 11 輸出層
    # 輸入為 上一層的輸出2048，輸出為class數目，在AlexNet中是1000
    model.add(Dense(512, nb_class, init='normal'))
    model.add(Activation('softmax'))
    return model
