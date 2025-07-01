import os


import cv2

from os.path import join
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from labellines import labelLine, labelLines
import numpy as np
from skimage import io, color
from statannot import add_stat_annotation
import pandas as pd
import seaborn as sns
import scipy

from lib_alban.utils_analyse import *
import math

import argparse


def predicted_chromaticity(imList, dataset):
    '''
    Function that loads full images and extracts the ref and match patches, predicted and GT, as well as their corresponding labels.
    Inputs:
        - imList: list of paths to images.
        - max_pxl: normalization
    Outputs:
        - GT, PRED, LABELS (order = )
    '''

    # Initialize outputs

    PRED = {}
    GT = {}
    LUM = {}
    LABELS = {}
    COUNT = {}
    conditions = ['normal', 'cube', 'floor', 'sphere', 'floorsphere', 'whole']
    shape = (len(imList)//len(conditions), 2)
    for condition in conditions:
        nb_labels = 5 # 0 is leftright, 1 is ref reflectance, 2 is ref lighting, 3 is match reflectance, 4 is match lighting
        PRED[condition] = np.zeros(shape); GT[condition] = np.zeros(shape); LUM[condition] = np.zeros(shape); LABELS[condition] = np.zeros((len(imList)//len(conditions), nb_labels))
        COUNT[condition] = 0
    for count, im in enumerate(imList):
        decomp_name = im.split('/')[-1].split('_')
        decomp_name[-1] = decomp_name[-1][:-4]  # extra character due to \n
        if len(decomp_name) == 4:
            condition = 'normal'
        else:
            condition = decomp_name[-1]

        for c, i in enumerate([1, 3, 4]):
            LABELS[condition][COUNT[condition], i] = decomp_name[c + 1]
            LABELS[condition][COUNT[condition], 3] = np.round(LABELS[condition][COUNT[condition], 3], 2)
        path2gt = join(path2predict, im[:-4] + '_ref.exr')
        path2pred = join(path2predict,im[:-4] + '_albedo0.exr')
        path2lum = join(path2predict, im[:-4] + '.exr')
        pred = load_and_process_exr(path2pred)
        gt = load_and_process_exr(path2gt)
        lum = load_and_process_exr(path2lum)[:,:,0]

        #print(int(decomp_name[2]))
        predleft, predright = mask_leftnright_patches(pred, maskleft_pred, maskright_pred )
        gtleft, gtright = mask_leftnright_patches(gt, maskleft_gt, maskright_gt)
        #import pdb; pdb.set_trace()
        lumleft, lumright = mask_leftnright_patches(lum, maskleft_gt, maskright_gt)
        pred_ref = np.median(predleft); pred_test = np.median(predright)
        gt_ref = np.median(gtleft); gt_test = np.median(gtright)
        lum_ref = np.median(lumleft); lum_test = np.median(lumright)

        PRED[condition][COUNT[condition], 0] = pred_ref; PRED[condition][COUNT[condition], 1] = pred_test
        GT[condition][COUNT[condition], 0] = gt_ref; GT[condition][COUNT[condition], 1] = gt_test
        LUM[condition][COUNT[condition], 0] = lum_ref; LUM[condition][COUNT[condition], 1] = lum_test
        COUNT[condition] += 1
    return GT, PRED, LUM, LABELS

def reshape_array(ARRAY, LABELS):
    '''
    Function that reshapes array of prediction or gt following the labels.
    Inputs:
        - ARRAY: shape = (3*21*2*5, 2)
    Outputs:
        - RESHAPED_ARRAY: shape = (3,34,2,2,5) = (ref, pred, leftright, refmatch, illus)
    '''

    # Initialize outputs
    shape = (len(list_labels_ref), len(list_labels_match), len(list_labels_leftright),2, len(list_labels_illu))
    RESHAPED_ARRAY = np.zeros(shape)
    for r in range(len(list_labels_ref)):
        for il in range(len(list_labels_illu)):
            for m in range(len(list_labels_match)):
                for lr in range(len(list_labels_leftright)):
                    toput = ARRAY[(LABELS[:, 1] == list_labels_ref[r]) & (LABELS[:, 4] == list_labels_illu[il]) & (LABELS[:, 3] == list_labels_match[m]) & (LABELS[:, 0] == list_labels_leftright[lr])]
                    RESHAPED_ARRAY[r, m, lr, :, il] = toput
    return RESHAPED_ARRAY

def compute_match(array_refs, array_matches, reference, luminance, labels, plot = True, labs = ['', 0,0,0]):
    '''
    Function that interpolates between the 2 closest predictions to find match.
    Inputs:
        - array_refs: array of pred for we want to match
        - array_matches: array of predictions for match to interpolate
        - labels: array(labels_matches)
    Outputs:
        - match: scallar
    '''


    #ref = np.nanmean(array_refs)
    diff = (array_matches - array_refs)
    interp = scipy.interpolate.interp1d(diff, labels, fill_value="extrapolate")
    #reg = LinearRegression()
    #reg.fit(array_matches[:, np.newaxis], labels)
    match = interp(0)
    #match = reg.predict(ref.reshape(1, 1))
    if plot == True:
        x_new = np.linspace(diff.min(),diff.max())
        y_new = interp(x_new).T
        #y_new = reg.predict(x_new[:, np.newaxis])
        fig, subs = plt.subplots(1,1)
        subs.scatter(diff, labels, color = 'k')
        subs.scatter(0, match, color = 'orange')
        subs.scatter(0, reference, color='g')
        subs.scatter(0, luminance, color='r')
        subs.plot(x_new, y_new, color='b')
        plt.xlabel('Predicted difference')
        plt.ylabel('Ground truth reflectance')
        plt.title(labs)
        fig.savefig('figures/{0}/{1}/interpolations/{2}.png'.format(arch, savename, labs))
        #plt.show()
        plt.close()
    return match

####-------------------------------------------------------------------------------------------------------------------------
####-------------------------------------------------------------------------------------------------------------------------

### SCRIPT ------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--path2outs', default='TwoTables_AlbanNetsup_0' , help='path to test dataset')
parser.add_argument('--plots', default=False, help='whether to plot graphes or not')
parser.add_argument('--nb_instances', type=int, default=2)
opt = parser.parse_args()
print(opt)

## Architecture and dataset to analyse
arch = opt.path2outs.split('_')[1]
dataset = opt.path2outs.split('_')[0]
conditions = ['normal', 'cube', 'floor', 'sphere', 'floorsphere', 'whole']
### initialization
ALL_MATCHES = {} ; ALL_ILLUREF = {} ; ALL_ILLUTEST = {} ; ALL_REF = {}
for condition in conditions:
    ALL_MATCHES[condition] = []; ALL_ILLUREF[condition] = []; ALL_ILLUTEST[condition] = []; ALL_REF[condition] = []

## Paths dataset
if dataset == 'TwoFloorsBlendereevee2p9':
    path2dataset = '/home/alban/Documents/blender_testset/testset/images_floorpattern_eevee2p9'
elif dataset == 'TwoCubesBlendereevee2p9':
    path2dataset = '/home/alban/Documents/blender_testset/testset/images_cubespattern_eevee2p9'
elif dataset == 'TwoCubesBlender1282p9':
    path2dataset = '/home/alban/Documents/blender_testset/testset/images_cubespattern_1282p9'
elif dataset == 'TwoCubesBlendereevee2p9nogamma':
    path2dataset = '/home/alban/Documents/blender_testset/testset/images_cubespattern_eevee2p9_nogamma'
elif dataset == 'TwoCubesBlendereevee3p6old':
    path2dataset = "/home/alban/Documents/blender_testset/testset/images_4test"
elif dataset == 'TwoCubesBlendereevee2p9centernogamma':
    path2dataset = '/home/alban/Documents/blender_testset/testset/images_cubespattern_eevee2p9_nogamma'
elif dataset == 'TwoCubesBlendereevee3p6centernogamma':
    path2dataset = '/home/alban/Documents/blender_testset/testset/images_cubespattern_eevee3p6_centernogamma'


with open(join(path2dataset, 'imList_all.txt')) as f:
    imList = [l.rstrip() for l in f.readlines()]

list_labels_ref, list_labels_illu, list_labels_match, list_labels_leftright = labels(dataset)



for ins in range(1, opt.nb_instances):
    instance = ins


    ## Save directory
    savename = dataset + '_' + arch + '_' + str(instance)

    #path2predict = '/home/alban/Dropbox/mlab/users/alban/works/inverse rendring/projects/Li et al,2020/Tests/results_%s_brdf14'%savename
    path2predict = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/pipeline_intrinsic_3datasets/test_outs/results_%s'%savename
    #savename = dataset + '_' + arch + '_ALL'

    os.system('mkdir figures/{0}'.format(arch))
    os.system('mkdir figures/{0}/{1}'.format(arch, savename))
    os.system('mkdir figures/{0}/{1}/interpolations'.format(arch,savename))


    ## Coords for color extraction
    maskleft_gt, maskright_gt, maskleft_pred, maskright_pred = def_masks_extract(dataset, arch)

    ## Extract colors and labels
    GT, PRED, LUM, LABELS = predicted_chromaticity(imList, dataset)

    ## COMPUTE MATCHING WITH INTERPOLATION

    ## RESHAPE ARRAYs

    for condition in conditions:
        PRED[condition] = reshape_array(PRED[condition], LABELS[condition])
        GT[condition] = reshape_array(GT[condition], LABELS[condition])
        LUM[condition] = reshape_array(LUM[condition], LABELS[condition])
        LABELS[condition] = reshape_array(LABELS[condition][:,[1,3]], LABELS[condition])
    #LABELS = reshape_array(LABELS[:,[1,3]])

    MATCHES = {}
    BR = {}
    for condition in conditions:
        MATCHES[condition] = np.zeros((len(list_labels_ref), len(list_labels_leftright), len(list_labels_illu)))
        BR[condition] = np.zeros((len(list_labels_ref), len(list_labels_leftright), len(list_labels_illu)))
        RESHAPED_GTtmp = GT[condition].copy()
        RESHAPED_GTtmp[RESHAPED_GTtmp==0]=0.001
        RESHAPED_ILLUM = LUM[condition]/RESHAPED_GTtmp

        illumref = RESHAPED_ILLUM[:,1:,:,0].mean(1)
        illumtest = RESHAPED_ILLUM[:,1:,:,1].mean(1)

        rrefref = GT[condition][:,:,list_labels_leftright, 0].mean(1) # reference reflectance
        rlum = rrefref*(illumref[:,list_labels_leftright]/illumtest[:,list_labels_leftright]) # luminance


        for r in range(len(list_labels_ref)):
            for il in range(len(list_labels_illu)):
                for lr in range(len(list_labels_leftright)):
                    MATCHES[condition][r, lr, il] = compute_match(PRED[condition][r,:,lr,0,il], PRED[condition][r,:,lr,1,il], rrefref[r, lr, il], rlum[r, lr, il], GT[condition][r,:,lr,1,il], plot = True, labs = [condition, list_labels_ref[r], list_labels_leftright[lr], list_labels_illu[il]])
                    BR[condition][r, lr, il] = np.absolute(MATCHES[condition][r, lr, il] - rlum[r, lr, il])/np.absolute(rrefref[r, lr, il] - rlum[r, lr, il])
        MATCHES[condition][MATCHES[condition]<0] = 0.01

        ALL_REF[condition].append(rrefref)
        ALL_MATCHES[condition].append(MATCHES[condition])
        ALL_ILLUTEST[condition].append(illumtest)
        ALL_ILLUREF[condition].append(illumref)


        # Sanity check
        ''' fig, subs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))
        # data plots
        subs[0].scatter(LABELS[condition][:, :, :, 0], GT[condition][:, :, :, 0], color='k')
        subs[1].scatter(LABELS[condition][:, :, :, 1], GT[condition][:, :, :, 1], color='k')
        # Formating
        subs[0].set_title('Ref reflectance GT vs label')
        subs[1].set_title('Match reflectance GT vs label')
        subs[0].set_xlabel('Labels')
        subs[1].set_xlabel('Labels')
        subs[0].set_ylabel('GT')
        plt.tight_layout()
        fig.savefig('figures/%s/%s/sanitycheck_%s_%s.png' % (arch, savename, savename, condition))
        # plt.show()
        plt.close()'''

        gvalue = np.arange(0.2, 0.81, (0.8 - 0.2) / (len(list_labels_ref) - 1))
        colors_leftright = [tuple([i, i, i]) for i in gvalue]

        gvalue = np.arange(0.2, 0.81, (0.8 - 0.2) / (len(list_labels_ref) - 1))
        colors_ref = [tuple([i, i, i]) for i in gvalue]

        gvalue = np.arange(0.2, 0.81, (0.8 - 0.2) / (len(list_labels_illu) - 1))
        colors_illu = [tuple([i, i, i]) for i in gvalue]
        fig, subs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6, 4))
        # data plots

        for count, il in enumerate(list_labels_illu):
            subs[0].plot(GT[condition][:, :, :, 0, count].mean(axis=(1,-1)), PRED[condition][:, :, :, 0, count].mean(axis=(1,-1)),
                            color=colors_illu[count], linestyle = '-', lw = 0.7, marker='o')
            subs[1].plot(GT[condition][:, :, :, 1, count].mean((0,-1)), PRED[condition][:, :, :, 1, count].mean((0,-1)),
                            color=colors_illu[count], linestyle = '-', lw = 0.7, marker='o')
        # Formating
        subs[0].set_title('Reference', fontsize=18)
        subs[1].set_title('Test', fontsize=18)
        subs[0].set_xlabel('True albedo', fontsize=16)
        subs[1].set_xlabel('True albedo', fontsize=16)
        subs[0].set_ylabel('Estimated albedo', fontsize=16)
        plt.tight_layout()
        fig.savefig('figures/%s/%s/UpdatedPredsVSGTwIllu_%s_%s.png' % (arch, savename, savename, condition), dpi=600)
        # plt.show()
        plt.close()

    #import pdb; pdb.set_trace()
    #fig, sub = plt.subplots(1, 1)
    #dat = [np.mean(BR[condition][:,0, 1:]) for condition in conditions]
    #sub.scatter(conditions, dat)
    #plt.show()

## SAVE DATA
for condition in  conditions:
    ALL_MATCHES[condition] = np.array(ALL_MATCHES[condition]); ALL_ILLUREF[condition] = np.array(ALL_ILLUREF[condition]); ALL_ILLUTEST[condition] = np.array(ALL_ILLUTEST[condition]); ALL_REF[condition] = np.array(ALL_ILLUTEST[condition])

os.system('mkdir results_analysis/{0}'.format(arch))

savename = savename = dataset + '_' + arch + '_ALL'
with open('results_analysis/%s/matches_%s.pkl'%(arch, savename), "wb") as write_file:
    pickle.dump(ALL_MATCHES, write_file)
with open('results_analysis/%s/illutest_%s.pkl'%(arch, savename), "wb") as write_file:
    pickle.dump(ALL_ILLUTEST, write_file)
with open('results_analysis/%s/illuref_%s.pkl' % (arch, savename), "wb") as write_file:
    pickle.dump(ALL_ILLUREF, write_file)
with open('results_analysis/%s/rref_%s.pkl' % (arch, savename), "wb") as write_file:
    pickle.dump(ALL_REF, write_file)




