# coding=utf-8
import numpy as np
import cPickle
import cv2
from os.path import basename
from os.path import dirname
import scipy.cluster.vq as vq
import matplotlib.pyplot as plt

#from roi_crop_mouse import on_mouse, mouse_crop



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
    label_cnt = len(set(img.label for img in all_imgs))
    output = []
    for label in range(label_cnt):
        output.append([each for each in all_imgs if each.label == label])
    return output

def codebook_dispatch(img,cb_list,bownorm=False):
    '''
    img: A Landmark object img
    cb_list: A "ordered" list of each class codebook
    norm: Ture or False, look compute_bow() in landmark.py
    output: img.bow, and each class will get it's bow
    '''
    if bownorm == True:
        img.compute_bow(cb_list[img.label],bownorm=True)
    else:
        img.compute_bow(cb_list[img.label])
    return img.bow



class Image(object):
    def __init__(self, path, img=None, cname=None, sift=None, sift_kp=None, bow=None, bow_img_path=None, wbow=None, tfidf=None, label=None, plabel=None, pcname=None, info=None):
        self.path = path
        self.img = cv2.imread(self.path)
        self.cname = cname # Class name
        self.sift = sift
        self.sift_kp = sift_kp
        self.bow = bow # hist
        self.bow_img_path = bow_img_path # path to store bow hist img
        self.wbow = wbow # Tf-idf Weight * bow 
        self.tfidf = tfidf
        self.label = label # int
        self.plabel = plabel # int, Predicted label
        self.pcname = pcname # str, Predicted class name
        self.info = info # dict, a package of img information
    
    #def get_img(self):
    #    self.img = cv2.imread(self.path)

    def get_fname(self):
        fname = basename(self.path)
        return fname
    
    def assign_label(self):
        tmpdir = dirname(self.path)
        tmpfolder = basename(tmpdir)
        folder = tmpfolder.split('.')
        self.cname = folder[1]
        self.label = int(folder[0]) 
        return self.label
            
    def get_sift(self):
        # Mask function is in code version 1.0
        self.img = cv2.imread(self.path,0)
        sift = cv2.SIFT()
        kp, des = sift.detectAndCompute(self.img,None)
        self.sift_kp = kp
        self.sift = des
        return kp, des

    def compute_bow(self, codebook, bownorm=False,histnorm=True):
        '''
        Get a bag of visual word
        
        compute_bow(SIFT feature) --> bow
        bow type: (k,) ,a k-dimension numpy array, k is codebook.shape[0]
        '''
        if histnorm == False:
            bow = compute_hist(codebook,self.sift,norm=False)
            if bownorm == True:
                from sklearn.preprocessing import MinMaxScaler
                min_max_scaler = MinMaxScaler()
                bow = min_max_scaler.fit_transform(bow)
            self.bow = bow
        else: #histnorm == True
            bow = compute_hist(codebook,self.sift,norm=True)
            if bownorm == True:
                from sklearn.preprocessing import MinMaxScaler
                min_max_scaler = MinMaxScaler()
                bow = min_max_scaler.fit_transform(bow)
            self.bow = bow
        return self.bow

    def graph_bow_hist(self,bowimg=False):
        '''
        Graph bow hist image from self.bow
        Create a path to store this img
        Return this path to self.bow_img_path
        '''
        if bowimg == True:
            bin_num = self.bow.shape[0]
            x = range(bin_num)
            y = self.bow
            fig = plt.figure(figsize=(2, 2)) #(2,2) means 200x200
            plt.plot(x,y)
            plt.xlim(0,len(x))
            plt.ylim(0,0.05)
            plt.grid(True)
            path = '/Users/Cyrus/Documents/vim/landmark_project/report/ver3/hist_bow/' + 'hist_' + self.get_fname()
            self.bow_img_path = path
            fig.savefig(path)


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



    def draw_kp(self):
        #path = REPORTDATA + '/' + self.get_fname()
        path = '/Users/Cyrus/Documents/vim/landmark_project/report/ver3/kpimg/' + self.get_fname()
        kpimg=cv2.drawKeypoints(self.img, self.sift_kp)
        cv2.imwrite(path,kpimg)
        return path
        #cv2.imshow('img '+ str(self.id), img)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
    
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

