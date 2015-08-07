# coding=utf-8
import numpy as np
import cPickle
import cv2
from os.path import basename
from os.path import dirname
import scipy.cluster.vq as vq
import matplotlib.pyplot as plt
import random
from sklearn.cross_validation import train_test_split
#from roi_crop_mouse import on_mouse, mouse_crop

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def cnn_data_package2(all_imgs,train_size=0.7,random_state=0):
    """ IMPORTANT
    This method package data in RGB color to numpy 3d array format
    keras API use only

    Argument:
        type all_imgs: list
        param all_imgs: A list of landmark.Image object

        type train_size: int
        param train_size: Same as  param of train_test_split() in sklearn
    Return:

    """
    for each in all_imgs:
        down = down_sample(each.img,32)
        height, width = down.shape[0], down.shape[1]
        each.img_for_cnn = down.reshape(1,3,height,width)

    #split = split_img_by_class(all_imgs)

    lbs = [each.label for each in all_imgs]
    x_train_, x_test_, y_train_, y_test_ = train_test_split(all_imgs,lbs,train_size=train_size,random_state=random_state)

    labels = list(set(lbs))
    n_class = len(labels)

    x_train = np.asarray([each.img_for_cnn for each in x_train_])
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[2],x_train.shape[3],x_train.shape[4])

    y_train = []
    for l in y_train_:
        arr_label = np.zeros((n_class,))
        i = labels.index(l)
        arr_label[i] = 1
        y_train.append(arr_label)
    y_train = np.asarray(y_train)


    x_test = np.asarray([each.img_for_cnn for each in x_test_])
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[2],x_test.shape[3],x_test.shape[4])

    y_test = []
    for l in y_test_:
        arr_label = np.zeros((n_class,))
        i = labels.index(l)
        arr_label[i] = 1
        y_test.append(arr_label)
    y_test = np.asarray(y_test)

    return (x_train, y_train), (x_test, y_test)


def cnn_data_package(all_imgs):
    """ IMPORTANT
    This method package only gray level image to dim 1*n
    theano API use only

    Argument:
        type all_imgs: list
        param all_imgs: A list of landmark.Image object

    Return: A list of file name is ignored
    """
    for each in all_imgs:
        gray = cv2.cvtColor(each.img,cv2.COLOR_BGR2GRAY)
        each.img_for_cnn = down_sample(gray,128)

    split = split_img_by_class(all_imgs)

    tmp_train, tmp_valid, tmp_test = [], [], []
    for c in split:
        n_img = len(c)
        n_train, n_valid, n_test = 3*n_img/5, n_img/5, n_img/5
        diff = n_img - (n_train + n_valid + n_test)
        if diff != 0:
            n_train = n_train + diff
        train_, valid_, test_ = c[:n_train], c[n_train : n_train+n_valid], c[n_train+n_valid:]
        tmp_train.extend(train_)
        tmp_valid.extend(valid_)
        tmp_test.extend(test_)
        #random.sample(list(n_img), n_train)

    def package(data):
        array = [np.concatenate(row.img_for_cnn) / 256. for row in data]
        label = [each.label for each in data]
        fname = [str(each.get_fname()) for each in data]
        return tuple((np.asarray(array), np.asarray(label))), fname

    train, name_train = package(tmp_train)
    valid, name_valid = package(tmp_valid)
    test, name_test = package(tmp_test)
    name = [name_train, name_valid, name_test]

    return train, valid, test


def down_sample(img,size=256):
    """ Crop image to square and resize to n*n

    Argument:
        type img: numpy.array
        param img: image read from cv2.imread

        type size: int, e.g. 128, 256,... 2^k
        param szie: image size you want

    Return:
        type new_img: numpy.array
        param new_img: image with new shape and crop to square
    """
    new_size = (size,size)
    width, height = img.shape[1], img.shape[0]
    if width> height:
        w = (width- height) / 2
        crop_img = img[:,w:w + height]
    elif height > width:
        h = (height - width) / 2
        crop_img =img[h:h+height,:]
    else:
        is_squre = True
        crop_img = img
    crop_and_resize_img = cv2.resize(crop_img,new_size)
    return crop_and_resize_img




def build_codebook(feature, voc_size, output):
    '''
    build_codebook(feature_list, K, path) --> codebook file

    feature: A list inculde all features
    voc_size: K in K-means, k is also called vocabulary size
    output: Path to store it
    '''
    K_THRESH = 1e-5
    farray = np.vstack(feature)
    codebook, distortion = vq.kmeans(farray, voc_size,thresh=K_THRESH)
    with open(output, 'wb') as f:
        cPickle.dump(codebook, f, protocol=cPickle.HIGHEST_PROTOCOL)

    print "-> %s Codebook built" %(output)
    return codebook


def load_codebook(codebook_path):
    f = open(codebook_path)
    codebook = cPickle.load(f)
    f.close()
    return codebook


def compute_hist(codebook, descriptors,norm=True):
    code, dist = vq.vq(descriptors, codebook)
    if norm == True: # norm==True --> sum(bow) = 1
        histogram_of_words, bin_edges = np.histogram(code, bins=range(codebook.shape[0] + 1), normed=True)
    else: # norm = Fasle
        histogram_of_words, bin_edges = np.histogram(code, bins=range(codebook.shape[0] + 1), normed=False)
    return histogram_of_words


def bow_hist(bow, output=False):
    '''
    input: A bow return by compute_hist()
    output: if output==True,then save pic
    '''
    bin_num = bow.shape[0]
    x = range(bin_num)
    y = bow
    if output == True:
        fig = plt.figure(figsize=(2, 2)) #(2,2) means 200x200
        plt.plot(x,y)
        plt.xlim(0,len(x))
        plt.ylim(0,0.1)
        plt.grid(True)
        path = '/Users/Cyrus/Documents/vim/landmark_project/report/ver3/hist_bow/' + 'hist_' + self.get_fname()
        self.bow_img_path = path
        fig.savefig(path)
    else:
        print "not yet"

def split_img_by_class(all_imgs):
    '''
    input: A list of Landmark object img
    output: A list of each class
    '''
    label= list(set(img.label for img in all_imgs))
    #label_cnt[-1] = label_cnt[-1] + 1
    output = []
    for l in label:
        output.append([each for each in all_imgs if each.label == l])
    return output

def codebook_dispatch(img,feature,cb_list,bownorm=False):
    '''
    img: A Landmark object img
    feature: sift or gist for now
    cb_list: A "ordered" list of each class codebook
    norm: Ture or False, look compute_bow() in landmark.py
    output: img.bow, and each class will get it's bow
    '''
    if bownorm == True:
        img.compute_bow(feature,cb_list[img.label],bownorm=True)
    else:
        img.compute_bow(feature,cb_list[img.label])
    return img.bow


IMG_SIZE = 200

class Image(object):
    def __init__(self, path, down_sample=False, img=None, cname=None, sift_descriptors=None, sift_keypoints=None,
                gist_descriptor=None, dsift_descriptors=None,dsift_keypoints=None, code=None,
                label=None, plabel=None, pcname=None, info=None, inlier_count=0):

        self.path = path
        self.down_sample = down_sample
        if self.down_sample == True:
            self._down_sample_()
        else:
            self.img = cv2.imread(self.path)
        try:
            self.assign_label()
        except:
            print "--> Label can't be found"
        self.inlier_count = inlier_count # for SIFT/RANSAC matching after Gist clustering
        #self.cname = cname # Class name
        #self.bow = bow # hist
        #self.spbow = spbow # Spatial Pyramid Matching method
        #self.bow_img_path = bow_img_path # path to store bow hist img
        #self.wbow = wbow # Tf-idf Weight * bow
        #self.tfidf = tfidf
        #self.label = label # int
        #self.plabel = plabel # int, Predicted label
        #self.pcname = pcname # str, Predicted class name
        #self.info = info # dict, a package of img information

    def _down_sample_(self):
        img = cv2.imread(self.path)
        try:
            x, y = img.shape[0], img.shape[1]
            if x > y:
                ratio = round(float(x)/y,3)
                x = int(round(IMG_SIZE*ratio))
                y = IMG_SIZE
                new_size = (x,y)
            elif x < y:
                ratio = round(float(y)/x,3)
                y = int(round(IMG_SIZE*ratio))
                x = IMG_SIZE
                new_size = (y,x)
            else:
                x = IMG_SIZE
                y = IMG_SIZE
                new_size = (x,y)
            self.img = cv2.resize(img,new_size)
        except AttributeError:
            print '-- Error at %s' %(str(self.get_fname()))
            #self.img = 'Reduction Error'
            raise EOFError('Down Sampling failed')

    def resize_to_square(self,size=200):
        new_size = (size,size)
        width, height = self.img.shape[1], self.img.shape[0]
        if width> height:
            w = (width- height) / 2
            crop_img = self.img[:,w:w + height]
        elif height > width:
            h = (height - width) / 2
            crop_img = self.img[h:h+height,:]
        else:
            is_squre = True
            crop_img = self.img
        crop_and_resize_img = cv2.resize(crop_img,new_size)
        self.square_img= crop_and_resize_img
        return crop_and_resize_img


    def get_fname(self):
        fname = unicode(str(basename(self.path)),'utf-8')
        return fname


    def assign_label(self):
        tmpdir = dirname(self.path)
        tmpfolder = basename(tmpdir)
        try:
            folder = tmpfolder.split('.')
            self.cname = folder[1]
        except IndexError:
            pass
        self.label = int(folder[0])
        return self.label


    def assign_p_label(self,number):
        '''
        assign_p_label(5) --> self.plabel = 5
        number: int, predicted label from svm
        '''
        self.plabel = number

    def assign_p_class(self,classtable):
        '''
        classtable: Returned from label_relation(x) from classlabel.py
        datatype: dict
        {'1':'landmarkA',
        '2':'labdmarkB'}
        '''
        for i in range(len(classtable)):
            if self.plabel == i:
                self.pcname = classtable[str(i)]


    def get_info_package(self, kpimg=False):
        '''
        Build a report information automatically
        Data type: Dict
        Store it to self.report
        '''
        self.info = {"class_name":"",
                       "predict_name":"",
                       "fname":"",
                       "img":"",
                       "bow_img":"",
                       "kp_number":""
                       }

        self.info["class_name"] = self.cname
        self.info["predict_name"] = self.pcname
        self.info["fname"] = self.get_fname()
        self.info["img"] = self.path
        self.info["bow_img"] = self.bow_img_path
        self.info["kp_number"] = len(self.sift_kp)
        if kpimg == True:
            self.info["img"] = self.draw_kp()

    def imshow(self):
        img = cv2.imread(self.path)
        cv2.imshow(str(self.get_fname()),img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def flip_x(self):
        self.img = cv2.flip(self.img,1) # 1 means flipping around the x-axis

    #def draw_kp(self):
    #    #path = REPORTDATA + '/' + self.get_fname()
    #    path = '/Users/Cyrus/Documents/vim/landmark_project/report/ver3/kpimg/' + self.get_fname()
    #    kpimg = cv2.drawKeypoints(test.img,kp)
    #    cv2.imwrite(path,kpimg)
    #    return path
    #    #cv2.imshow('img '+ str(self.id), img)
    #    #cv2.waitKey()
    #    #cv2.destroyAllWindows()

    #def draw_roi(self,background):
    #    if background == 'RGB':
    #        tmp = cv2.rectangle(self.rgb, (self.roi[0], self.roi[1]), (self.roi[2], self.roi[3]), (0,255,0), 2)
    #        cv2.imshow(str(self.id),self.rgb)
    #    elif background == 'GRAY':
    #        tmp = cv2.rectangle(self.gray, (self.roi[0], self.roi[1]), (self.roi[2], self.roi[3]), (0,255,0), 2)
    #        cv2.imshow(str(self.id),self.gray)
    #    elif background == "BW":
    #        tmp = cv2.rectangle(self.bw, (self.roi[0], self.roi[1]), (self.roi[2], self.roi[3]), (0,255,0), 2)
    #        cv2.imshow(str(self.id),self.bw)
    #    cv2.waitKey()
    #    cv2.destroyAllWindows

    #def create_mask(self):
    #    '''
    #    mask就是一個和源影像一樣大小的矩陣，
    #    roi的地方擺255，其他擺0，
    #    注意後面要加上np.unit8才符合CV_U8的格式
    #    '''
    #    mask = np.zeros(self.gray.shape, np.uint8)
    #    mask[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]] = 255
    #    return mask
