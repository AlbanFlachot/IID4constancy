'''
Script to convert a set of images in EXR to PNG
Starts by opoening all images and saving the max and min values.
We will use this values to normalize all images
'''


import os

import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from os.path import join
import matplotlib.pyplot as plt
import glob

def load_and_process_exr(path, gamma = False):
    '''
    Loads an hdr/exr image and preprocesses it
    '''
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # load image
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == 'uint8':
        img = img / 255
    #img = np.clip(img, 0,1)
    if gamma == True:
        img = img**(2.2)
    return img

def split3channels(img):
    '''
    Split an RGB channels into 3 images, one each channel
    '''

    img_B = img[:,:,0] # cv2 loads in BGR
    img_G = img[:,:,1]
    img_R = img[:,:,2]
    return img_R, img_G, img_B

def gammacorr(img):
    '''
    Function to do gammacorrection. img assumed to be float between 0 and 1
    '''

    return img**(1/2.2)

dirpath = "/home/alban/Documents/blender_testset/testset/"
dataset = 'images_cubespattern_1282p9'

path2dataset = join(dirpath, dataset)

list_images = glob.glob(path2dataset + '/*_ref.exr')
list_images = [x.replace('_ref.exr', '.exr') for x in list_images]

print('Computing min and max values of the EXR dataset')
'''for imgp in list_images:
    img = load_and_process_exr(imgp)
    maxv = 0; minv = 0
    maxv = max(maxv, np.max(img))
    #maxv = max(maxv, np.percentile(img, 99))
    minv = min(minv, np.min(img))'''

maxv =1; minv=0

print(f'The min and max pxl values are [{minv}, {maxv}]')

print(('Comverting EXR to PNGs using min & max values'))

path2dirsave = join(dirpath, dataset + '_PNGgamma4paperresized')
if not os.path.exists(path2dirsave):
    os.mkdir(path2dirsave)

#import pdb; pdb.set_trace()
for imgp in list_images:
    img = load_and_process_exr(imgp)
    imgpng =(img - minv)/(maxv-minv)
    imgpng[imgpng>1]=1
    imgpng=gammacorr(imgpng)
    #import pdb; pdb.set_trace()
    imgpngresized = (255*cv2.resize(imgpng, (512,512), interpolation=cv2.INTER_NEAREST)).astype(int)
    imgpng_R, imgpng_G, imgpng_B= split3channels(imgpngresized)
    imgname = imgp.split('/')[-1][:-4]
    cv2.imwrite(join(path2dirsave, imgname + '_R.png'), imgpng_R)
    cv2.imwrite(join(path2dirsave, imgname + '_G.png'), imgpng_G)
    cv2.imwrite(join(path2dirsave, imgname + '_B.png'), imgpng_B)


