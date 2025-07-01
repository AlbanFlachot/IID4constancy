import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import matplotlib.pyplot as plt
from os.path import join

def CCI(pred, illushift, D65_chrom):
    CCI = np.linalg.norm(pred - D65_chrom, axis = -1)
    CCI = 1 - CCI/np.linalg.norm(illushift, axis = -1)
    return CCI

def BR(pred, illushift, D65_chrom):
    BR = np.sum((pred- D65_chrom) * illushift, axis = -1)
    BR = BR/np.linalg.norm(illushift, axis = -1)
    return BR

def DeltaE(pred, gt):
    diff = pred - gt
    DeltaL = diff[:,:,0]
    Deltaab = diff[:, :, 1:]
    DeltaE = np.linalg.norm(diff, axis = -1)
    return DeltaE, DeltaL, Deltaab
    
def crop_leftnright_patches(img, coords_left, coords_right):
    "Function that extracts the cromaticity of the left and right patches "

    left = img[coords_left[0,0]:coords_left[0,1], coords_left[1,0]:coords_left[1,1]]
    right = img[coords_right[0, 0]:coords_right[0, 1], coords_right[1, 0]:coords_right[1, 1]]

    img2 = img.copy()
    img2[coords_left[0, 0]:coords_left[0, 1], coords_left[1, 0]:coords_left[1, 1]] = np.array([1,0,0])
    img2[coords_right[0, 0]:coords_right[0, 1], coords_right[1, 0]:coords_right[1, 1]] = np.array([1,0,0])

    fig, subs = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(6,4))
    subs[0].imshow(img2)
    #subs[0].imshow(img2)
    subs[1].imshow(left)
    subs[2].imshow(right)
    plt.yticks([])
    plt.tight_layout()
    #fig.savefig('figures/UnsIlluwLeftnRight.png')
    plt.show()
    plt.close()
    return left, right

def mask_leftnright_patches(img, mask_left, mask_right):
    "Function that extracts the cromaticity of the left and right patches "
    img2 = img.copy()
    img2[mask_left] = 0
    img2[mask_right] = 0
    if (mask_left.shape[-1] != img.shape[-1]):
        mask_left = np.stack((mask_left, mask_left, mask_left), axis = -1)
        mask_right = np.stack((mask_right, mask_right, mask_right), axis=-1)
    '''fig, subs = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(6,4))
    left = img*(mask_left*1)
    right = img*(mask_right*1)
    imgmax = img.max()
    subs[0].imshow(img2/imgmax)
    #subs[0].imshow(img2)
    subs[1].imshow(left/imgmax)
    subs[2].imshow(right/imgmax)
    plt.yticks([])
    plt.tight_layout()
    #fig.savefig('figures/UnsIlluwLeftnRight.png')
    plt.show()
    plt.close()'''

    return img[mask_left], img[mask_right]

def load_and_process(path):
    '''
    Loads and preprocesses the image
    '''
    img = cv2.imread(path)  # load image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == 'uint8':
        img = img / 255  # normalize it
    return img

def load_and_process_illusions(path):
    '''
    Loads and preprocesses the image
    '''
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # load image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == 'uint8':
        img = img / 255
    img = np.clip(img/img.max(), 0,1)
    return img

def load_and_process_exr(path, gamma = False):
    '''
    Loads and preprocesses the image
    '''
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # load image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == 'uint8':
        img = img / 255
    #img = np.clip(img, 0,1)
    if gamma == True:
        img = img**(2.2)
    return img


def def_masks_extract(dataset, arch):
    if 'old' in dataset:
        maskleft_pred = cv2.imread('/home/alban/Documents/blender_testset/testset/masks_nov2/leftsheet_reduced_ref.exr',
                                   cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).mean(-1)
        maskright_pred = cv2.imread(
            '/home/alban/Documents/blender_testset/testset/masks_nov2/rightsheet_reduced_ref.exr',
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).mean(-1)
    elif 'center' in dataset:
        maskleft_pred = cv2.imread('/home/alban/Documents/blender_testset/testset/masks_cubespattern_center/leftsheet_ref.exr',
                                   cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).mean(-1)
        maskright_pred = cv2.imread(
            '/home/alban/Documents/blender_testset/testset/masks_cubespattern_center/rightsheet_ref.exr',
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).mean(-1)
    elif 'Floor' in dataset:
        maskleft_pred = cv2.imread('/home/alban/Documents/blender_testset/testset/masks_floorpattern/leftsheet_ref.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).mean(-1)
        maskright_pred = cv2.imread('/home/alban/Documents/blender_testset/testset/masks_floorpattern/rightsheet_ref.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).mean(-1)
    elif 'Cubes' in dataset:
        maskleft_pred = cv2.imread('/home/alban/Documents/blender_testset/testset/masks_cubespattern/leftsheet_ref.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).mean(-1)
        maskright_pred = cv2.imread('/home/alban/Documents/blender_testset/testset/masks_cubespattern/rightsheet_ref.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).mean(-1)

    maskleft_pred[maskleft_pred<1] = 0 #use a fairly restrictive treshold to avoid edges
    maskleft_pred = maskleft_pred.astype(bool)
    maskright_pred[maskright_pred < 1] = 0
    maskright_pred = maskright_pred.astype(bool)
    maskleft_gt = maskleft_pred.copy() # there are no change in image size between input and output, thus masks are the same
    maskright_gt = maskright_pred.copy()
    return maskleft_gt, maskright_gt, maskleft_pred, maskright_pred

def labels(dataset):
    list_labels_ref = np.round(np.array([0.2, 0.4, 0.6]), 1)
    list_labels_illu = np.round(np.array([0, 0.35, 0.75, 1.5, 3]), 2)
    list_labels_match = np.array([0.05,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    list_labels_leftright = [0]
    return list_labels_ref, list_labels_illu, list_labels_match, list_labels_leftright




