# coding=utf-8
from os.path import exists
from os.path import isdir
from os.path import basename 
from os.path import join
from os.path import splitext
from glob import glob

EXTENSIONS = [".jpg", ".jpeg", "png", "pgm", "tif", "tiff"]

def get_categories(datasetpath):
    cat_paths = [files
                 for files in glob(datasetpath + "/*")
                  if isdir(files)]
    cat_paths.sort()
    cats = [basename(cat_path) for cat_path in cat_paths]
    return cats

def get_imgfiles(path):
    '''
    input: /datasetpath/folder/ 
    output: /datasetpath/folder1/.jpg , /datasetpath/folder2/.jpg
    '''
    all_files = []
    all_files.extend([join(path, basename(fname))
                    for fname in glob(path + "/*")
                    if splitext(fname)[-1].lower() in EXTENSIONS])
    return all_files

def get_all_imgfiles(datasetpath):
    all_folders = get_categories(datasetpath) 
    num_of_folder = len(all_folders)
    print "searching for folders at " + datasetpath
    if num_of_folder < 1:
        raise ValueError('Only ' + str(num_of_folder) + ' categories found. Wrong path?')
    print "found following folders / categories:"
    print all_folders
    print "---------------------" 
    all_img_files = []
    for folder in all_folders:
        path = join(datasetpath,folder)
        current_file = get_imgfiles(path)
        all_img_files.extend(current_file)
    return all_img_files

