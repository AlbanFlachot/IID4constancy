import os


import cv2

import pickle
import scipy

from lib_alban.utils_analyse import *
import math

import argparse


def predicted_chromaticity(imList, dataset, max_pxl = 255):
    '''
    Function that loads full images and extracts the ref and match patches, predicted and GT, as well as their corresponding labels.
    Inputs:
        - imList: list of paths to images.
        - max_pxl: normalization
    Outputs:
        - GT, PRED, LABELS (order = )
    '''

    # Initialize outputs
    shape = (len(imList), 2)
    nb_labels = 5 # 0 is leftright, 1 is ref reflectance, 2 is ref lighting, 3 is match reflectance, 4 is match lighting
    PRED = np.zeros(shape); GT = np.zeros(shape); LUM = np.zeros(shape); LABELS = np.zeros((len(imList), nb_labels))
    COUNT = 0
    for count, im in enumerate(imList):
        #print(im)
        if "TwoCubesBlender" in dataset:
            decomp_name = im.split('/')[-1].split('_')
            decomp_name[-1] = decomp_name[-1][:-4]  # extra character due to \n

            for c, i in enumerate([1, 3, 4]):
                LABELS[COUNT, i] = decomp_name[c + 1]
                LABELS[COUNT, 3] = np.round(LABELS[COUNT, 3], 2)
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

        PRED[COUNT, 0] = pred_ref; PRED[COUNT, 1] = pred_test
        GT[COUNT, 0] = gt_ref; GT[COUNT, 1] = gt_test
        LUM[COUNT, 0] = lum_ref; LUM[COUNT, 1] = lum_test
        COUNT += 1
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
    #import pdb; pdb.set_trace()
    RESHAPED_ARRAY = np.zeros(shape)
    for r in range(len(list_labels_ref)):
        for il in range(len(list_labels_illu)):
            for m in range(len(list_labels_match)):
                for lr in range(len(list_labels_leftright)):
                    #print([r, il, m, lr])
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
### initialization
ALL_MATCHES = [] ; ALL_ILLUREF = [] ; ALL_ILLUTEST = [] ; ALL_REF = []

for ins in range(1, opt.nb_instances):
    instance = ins

    ## Paths dataset
    if dataset == 'TwoCubesBlender1024':
        path2dataset = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/blender_1024_nov'
    elif dataset == 'TwoCubesBlender128':
        path2dataset = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/blender_128_nov'
    elif dataset == 'TwoCubesBlender1024nov2':
        path2dataset = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/blender_128_nov2'
    elif dataset == 'TwoCubesBlender128nov2':
        path2dataset = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/blender_128_nov2'
    elif dataset == 'TwoCubesBlendereevee':
        path2dataset = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/blender_eevee_nov'
    elif dataset == 'TwoCubesBlendereeveenov2':
        path2dataset = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/blender_eevee_nov2'
    elif dataset == 'TwoCubesBlender128Tom':
        path2dataset = '/home/alban/Documents/blender_testset/testset/images_128_Tom'
    elif dataset == 'TwoCubesBlender1024Tom':
        path2dataset = '/home/alban/Documents/blender_testset/testset/images_1024_Tom'
    elif dataset == 'TwoCubesBlendereeveeTom':
        path2dataset = '/home/alban/Documents/blender_testset/testset/images_eevee_Tom'
    elif dataset == 'TwoCubesBlender1283p6':
        path2dataset = '/home/alban/Documents/blender_testset/testset/images_2cubes_1283p6'
    elif dataset == 'TwoCubesBlendereevee3p6':
        path2dataset = '/home/alban/Documents/blender_testset/testset/images_2cubes_eevee3p6'
    elif dataset == 'TwoCubesBlenderfloor1283p6':
        path2dataset = '/home/alban/Documents/blender_testset/testset/images_floorpattern_1283p6'
    elif dataset == 'TwoCubesBlenderflooreevee3p6':
        path2dataset = '/home/alban/Documents/blender_testset/testset/images_floorpattern_eevee3p6'

    with open(join(path2dataset, 'imList_normal.txt')) as f:
        imList = [l.rstrip() for l in f.readlines()]

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

    list_labels_ref, list_labels_illu, list_labels_match, list_labels_leftright = labels(dataset)


    ## RESHAPE ARRAYs

    PRED = reshape_array(PRED, LABELS)
    GT = reshape_array(GT, LABELS)
    LUM = reshape_array(LUM, LABELS)
    LABELS = reshape_array(LABELS[:,[1,3]], LABELS)
    #LABELS = reshape_array(LABELS[:,[1,3]])

    MATCHES = {}
    BR = {}
    MATCHES = np.zeros((len(list_labels_ref), len(list_labels_leftright), len(list_labels_illu)))
    BR = np.zeros((len(list_labels_ref), len(list_labels_leftright), len(list_labels_illu)))
    RESHAPED_GTtmp = GT.copy()
    RESHAPED_GTtmp[RESHAPED_GTtmp==0]=0.001
    RESHAPED_ILLUM = LUM/RESHAPED_GTtmp

    illumref = RESHAPED_ILLUM[:,1:,:,0].mean(1)
    illumtest = RESHAPED_ILLUM[:,1:,:,1].mean(1)

    rrefref = GT[:,:,list_labels_leftright, 0].mean(1) # reference reflectance
    rlum = rrefref*(illumref[:,list_labels_leftright]/illumtest[:,list_labels_leftright]) # luminance



    for r in range(len(list_labels_ref)):
        for il in range(len(list_labels_illu)):
            for lr in range(len(list_labels_leftright)):
                MATCHES[r, lr, il] = compute_match(PRED[r,:,lr,0,il], PRED[r,:,lr,1,il], rrefref[r, lr, il], rlum[r, lr, il], GT[r,:,lr,1,il], plot = True, labs = ['normal', list_labels_ref[r], list_labels_leftright[lr], list_labels_illu[il]])
                BR[r, lr, il] = np.absolute(MATCHES[r, lr, il] - rlum[r, lr, il])/np.absolute(rrefref[r, lr, il] - rlum[r, lr, il])
    MATCHES[MATCHES<0] = 0.01

    ALL_REF.append(rrefref)
    ALL_MATCHES.append(MATCHES)
    ALL_ILLUTEST.append(illumtest)
    ALL_ILLUREF.append(illumref)

    # Sanity check
    fig, subs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))
    # data plots
    subs[0].scatter(LABELS[:, :, :, 0], GT[:, :, :, 0], color='k')
    subs[1].scatter(LABELS[:, :, :, 1], GT[:, :, :, 1], color='k')
    # Formating
    subs[0].set_title('Ref reflectance GT vs label')
    subs[1].set_title('Match reflectance GT vs label')
    subs[0].set_xlabel('Labels')
    subs[1].set_xlabel('Labels')
    subs[0].set_ylabel('GT')
    plt.tight_layout()
    fig.savefig('figures/%s/%s/sanitycheck_%s_%s.png' % (arch, savename, savename, 'normal'))
    # plt.show()
    plt.close()

    gvalue = np.arange(0.2, 0.81, (0.8 - 0.2) / (len(list_labels_ref) - 1))
    colors_leftright = [tuple([i, i, i]) for i in gvalue]

    gvalue = np.arange(0.2, 0.81, (0.8 - 0.2) / (len(list_labels_ref) - 1))
    colors_ref = [tuple([i, i, i]) for i in gvalue]

    gvalue = np.arange(0.2, 0.81, (0.8 - 0.2) / (len(list_labels_illu) - 1))
    colors_illu = [tuple([i, i, i]) for i in gvalue]
    fig, subs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))
    # data plots

    for count, il in enumerate(list_labels_illu):
        subs[0].scatter(GT[:, :, :, 0, count].mean(-1), PRED[:, :, :, 0, count].mean(-1),
                        color=colors_illu[count])
        subs[1].scatter(GT[:, :, :, 1, count].mean(-1), PRED[:, :, :, 1, count].mean(-1),
                        color=colors_illu[count])
    # Formating
    subs[0].set_title('Reference', fontsize=17)
    subs[1].set_title('Test', fontsize=17)
    subs[0].set_xlabel('Ground-truth', fontsize=14)
    subs[1].set_xlabel('Ground-truth', fontsize=14)
    subs[0].set_ylabel('Prediction', fontsize=14)
    plt.tight_layout()
    fig.savefig('figures/%s/%s/UpdatedPredsVSGTwIllu_%s_normal.png' % (arch, savename, savename))
    # plt.show()
    plt.close()

    #import pdb; pdb.set_trace()
    #fig, sub = plt.subplots(1, 1)
    #dat = [np.mean(BR[condition][:,0, 1:]) for condition in conditions]
    #sub.scatter(conditions, dat)
    #plt.show()




