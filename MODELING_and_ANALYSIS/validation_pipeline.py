import os
import cv2

import sys

import matplotlib.pyplot as plt

sys.path.append("/LiNet")
import torch
import numpy as np
from torch.autograd import Variable
import argparse
import random
from lib_alban.utils_analyse import def_masks_extract
import os.path as osp
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import lib_alban.utils_test as utils_test
from lib_alban.utils_analyse import load_and_process, load_and_process_exr, load_and_process_illusions


parser = argparse.ArgumentParser()
# The locationi of testing set
parser.add_argument('--testset', help='path to real images')
parser.add_argument('--imList', help='path to image list')
# Model
parser.add_argument('--arch', default='AlbanNet', help='the architecture')
parser.add_argument('--checkpointdir', default=None, help='the path to the checkpoint')
# Paths
parser.add_argument('--testRoot', help='the path to save the testing errors')
parser.add_argument( '--imagedir_out', type=str, default='../test_out/' )

parser.add_argument('--nepoch0', type=int, default=14, help='the number of epoch for testing')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for testing network')
parser.add_argument('--level', type=int, default=2, help='the cascade level')
parser.add_argument('--nb_channels', type=int, default=1, help='number of color channels')
# for IRNet
parser.add_argument( '--device',       type=str, default='cuda:0', help='device (cuda:0 or cpu)' )
# for LiNet
parser.add_argument( '--gamma',       type=int, default=0, help='if images need gamma correction 1 = Yes, 0 = No, -1 = invgamma for albedo' )

# Image Picking
opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]

nepochs = [opt.nepoch0]


opt.seed = 0
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)

opt.batchSize = 8
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if 'AlbanNet' in opt.arch:
    rootdir = 'AlbanNet/'
    Net = utils_test.AlbanNet(checkpointdir=opt.checkpointdir, cuda=opt.cuda, epoch = opt.nepoch0, nb_channels = opt.nb_channels)

Net.set4testing()


if opt.testset == 'validation_128':
    dataRoot = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/validation_128'
elif opt.testset == 'validation_1024':
    dataRoot = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/validation_1024'
elif opt.testset == 'validation_eevee':
    dataRoot = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/validation_eevee'
elif opt.testset == 'validation_geo_eevee':
    dataRoot = '/home/alban/Documents/val_geo_eevee'
elif opt.testset == 'TwoCubesBlender128':
    dataRoot = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/blender_128'
elif opt.testset == 'TwoCubesBlender512':
    dataRoot = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/blender_512'
elif opt.testset == 'TwoCubesBlender1024':
    dataRoot = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/blender_1024'
elif opt.testset == 'TwoCubesBlendereevee':
    dataRoot = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/blender_eevee'
elif opt.testset == 'TwoCubesBlender128nov2':
    dataRoot = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/blender_128_nov2'
elif opt.testset == 'TwoCubesBlender512nov2':
    dataRoot = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/blender_512_nov2'
elif opt.testset == 'TwoCubesBlender1024nov2':
    dataRoot = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/blender_1024_nov2'
elif opt.testset == 'TwoCubesBlendereeveenov2':
    dataRoot = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/blender_eevee_nov2'
elif opt.testset == 'TwoCubesBlendereeveepatterns':
    #dataRoot = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/blender_eevee_patterns'
    dataRoot = '/home/alban/Documents/blender_testset/testset/images_4test'
elif opt.testset == 'TwoCubesBlender128patterns':
    dataRoot = '/home/alban/Documents/blender_testset/testset/images_4test_128'
elif opt.testset == 'TwoCubesBlender1283p6patterns':
    dataRoot = '/home/alban/Documents/blender_testset/testset/images_4test_1283p6'
elif opt.testset == 'validation_1283p6patterns':
    dataRoot = '/home/alban/Documents/validation_1283p6'

### directory where to save 10 first outputs and inputs
os.system(f'mkdir {dataRoot}/outputs')

with open(os.path.join(dataRoot,  opt.imList), 'r') as imIdIn:
    imIds = imIdIn.readlines()
#imList = [osp.join(dataRoot, 'images/' + x.strip()) for x in imIds]
imList = [osp.join(dataRoot,  x.strip()) for x in imIds]
imList = sorted(imList)

PREDS = []
GT = []
LUM = []
IMSCORE = []
IMCORR = []

channel = 0
j = 0
quotient = len(imList)//opt.batchSize
modulo = len(imList)%opt.batchSize
if bool(modulo):
    nb_iter = quotient + 1 # in case modlo is not zero, we need to include one additional iteration
else:
    nb_iter = quotient
for iter in range(nb_iter):
    # if ("hdr_mondrian" not in imName):
    #    continue
    if iter==nb_iter-1:
        imNames = imList[iter * opt.batchSize:]  # this is mostly for the case where modulo not 0
    else:
        imNames = imList[iter * opt.batchSize:(iter + 1) * opt.batchSize]  # create our batch
    # if ("hdr_mondrian" not in imName):
    #    continue
    j+=1
    print(j)
    print('%d/%d' % (j, nb_iter))

    imOutputNames = []
    for imName in imNames:
        imId = imName.split('/')[-1]
        print(imId)
        imOutputNames.append(osp.join(f'{dataRoot}/outputs', imId))

    # compiling the batch
    batch = []
    batch_albedo = []
    for im_name in imNames:
        # Load the image from cpu to gpu

        assert (osp.isfile(im_name))
        if opt.testset == 'illusions':
            batch.append(load_and_process_illusions(im_name))
        else:
            batch.append(load_and_process_exr(im_name))
        batch_albedo.append(load_and_process_exr(im_name.replace('.exr', '_ref.exr')))

    # inference
    if 'AlbanNet' in opt.arch:
        albedoPreds, illuPreds = Net.runtest(batch)

    ### compute comparison and save

    idx = np.random.choice(range(256), 1000)
    idy = np.random.choice(range(256), 1000)

    ### put the input in the right format
    for i, img in enumerate(batch):
        batch[i] = np.transpose(img.cpu().numpy(), [1, 2, 0])
    for i, pred in enumerate(albedoPreds):
        albedoPred = pred
        PREDS.append(albedoPred[idx, idy,channel])
        GT.append(batch_albedo[i][idx, idy,channel])
        LUM.append(batch[i][idx, idy,channel])

        IMSCORE.append(np.round(np.median(np.absolute(albedoPred - batch_albedo[i][:,:,channel])), 3))
        IMCORR.append(np.round(np.corrcoef(albedoPred[:,:,channel].flatten(), batch_albedo[i][:,:,channel].flatten())[0, 1], 3))

        if j*opt.batchSize <11: # 10 first images

            cv2.imwrite(imOutputNames[i][:-4] + '_albedo0.exr', albedoPred.astype("float32"))
            cv2.imwrite(imOutputNames[i][:-4] + '.exr', batch[i][:,:,0].astype("float32"))
            cv2.imwrite(imOutputNames[i][:-4] + '_ref.exr', batch_albedo[i][:,:,0].astype("float32"))

#### compute accuracy accros images

PREDS = np.array(PREDS)
GT = np.array(GT)
LUM = np.array(LUM)



score = np.round(np.median(np.absolute(PREDS - GT)),3)
score_lum = np.round(np.median(np.absolute(LUM - GT)),3)
corr = np.round(np.corrcoef(PREDS.flatten(), GT.flatten())[0,1],3)
corr_lum = np.round(np.corrcoef(LUM.flatten(), GT.flatten())[0,1],3)
#import pdb; pdb.set_trace()

os.system('mkdir figures/{0}'.format(opt.arch))

fig, subs = plt.subplots(1,1,figsize = (4.3,4))
subs.scatter(GT[:1000, :10].flatten(), PREDS[:1000, :10].flatten(), s=5, color = 'k', alpha = 0.5)
#subs.scatter(GT.flatten(), PREDS.flatten(), s=0.05, color = 'k', alpha = 0.002)
#subs.scatter(GT.flatten(), LUM.flatten(), s=1, color = 'k', alpha = 0.002)
subs.set_xlim(0,1)
subs.set_ylim(0,1)
subs.plot([0,1],[0,1], 'k', lw = 1, ls = '--')
subs.set_xlabel("True albedo", fontsize = 15)
subs.set_ylabel("Estimated albedo", fontsize = 15)
#subs.text(0.05, 0.9, 'error = {:.3f}'.format(score), color = 'k', fontsize = 13)
#subs.text(0.05, 0.83, 'null = {:.3f}'.format(score_lum), color = 'k', fontsize = 12)
#subs.text(0.05, 0.83, 'r = {:.3f}'.format(corr), color = 'k', fontsize = 13)
subs.text(0.05, 0.9, 'error = {:.3f}'.format(score), color = 'k', fontsize = 12)
subs.text(0.05, 0.83, 'correlation = {:.3f}'.format(corr), color = 'k', fontsize = 12)
plt.tight_layout()
#plt.show()
fig.savefig('figures/{0}/{1}_{2}.png'.format(opt.arch, opt.arch, opt.testset), dpi = 500, )

fig, subs = plt.subplots(1,1,figsize = (4.3,4))
#subs.scatter(GT.flatten(), LUM.flatten(), s=0.05, color = 'k', alpha = 0.002)
subs.scatter(GT[:1000, :10].flatten(), LUM[:1000, :10].flatten(), s=5, color = 'k', alpha = 0.5)
subs.set_xlim(0,1)
subs.set_ylim(0,1)
subs.plot([0,1],[0,1], 'k', lw = 1)
subs.set_xlabel("True albedo", fontsize = 15)
subs.set_ylabel("Estimated albedo", fontsize = 15)
subs.text(0.05, 0.9, 'error = {:.3f}'.format(score_lum), color = 'k', fontsize = 12)
subs.text(0.05, 0.83, 'correlation = {:.3f}'.format(corr_lum), color = 'k', fontsize = 12)
#subs.text(0.05, 0.83, 'corr = {:.3f}'.format(corr_lum), color = 'k', fontsize = 12)
#subs.text(0.5, 0.15, 'r = {:.3f}'.format(corr), color = 'k', fontsize = 12)
plt.tight_layout()
#plt.show()
fig.savefig('figures/{0}/{1}_{2}_baseline.png'.format(opt.arch, opt.arch, opt.testset), dpi = 500, )

fig, subs = plt.subplots(1,2,figsize = (4.3,4))
subs[0].hist(IMSCORE)
subs[1].hist(IMCORR)
subs[0].set_ylabel("count", fontsize = 15)
subs[0].set_xlabel("error", fontsize = 15)
subs[1].set_xlabel("correlation", fontsize = 15)
#subs.text(0.5, 0.15, 'r = {:.3f}'.format(corr), color = 'k', fontsize = 12)
plt.tight_layout()
#plt.show()


print(score)


