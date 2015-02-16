# coding=utf-8
from gettingFile import get_categories
import json


# Build a relationship between class and label
# eg. 'tpe101':'1' , 'mall':'2'
# Use a json file store it
def label_relation(filepath):
    '''
    filepath: e.g. data/ver3data/test_data
    return: {'1':'landmarkA',
             '2':'labdmarkB'}
    '''
    folders = get_categories(filepath)
    class_label = {}
    tmp = [f.split('.') for f in folders]
    for x in tmp:
        class_label[x[1]] = x[0]
    #print str(class_label).decode('string_escape')
    label_inverse = {val:key for key,val in class_label.items()}
    return label_inverse
    #with open(DATASETPATH + '/' + 'classlabel.json', 'w') as f:
    #    json.dump(label_inverse, f)
