import sys
sys.path.append("/LiNet")
import torch
import numpy as np
from torch.autograd import Variable
import argparse
import random
import os
import os.path as osp
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import lib_alban.utils_test as utils_test
from lib_alban.utils_analyse import load_and_process, load_and_process_exr, load_and_process_illusions
import imageio

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

# The basic testing setting
parser.add_argument('--nepoch0', type=int, default=14, help='the number of epoch for testing')

parser.add_argument('--imHeight0', type=int, default=240, help='the height / width of the input image to network')
parser.add_argument('--imWidth0', type=int, default=320, help='the height / width of the input image to network')
parser.add_argument('--imHeight1', type=int, default=240, help='the height / width of the input image to network')
parser.add_argument('--imWidth1', type=int, default=320, help='the height / width of the input image to network')

parser.add_argument('--envRow', type=int, default=120, help='the height /width of the envmap predictions')
parser.add_argument('--envCol', type=int, default=160, help='the height /width of the envmap predictions')
parser.add_argument('--envHeight', type=int, default=8, help='the height /width of the envmap predictions')
parser.add_argument('--envWidth', type=int, default=16, help='the height /width of the envmap predictions')


parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for testing network')

parser.add_argument('--level', type=int, default=2, help='the cascade level')
parser.add_argument('--nb_channels', type=int, default=1, help='number of color channels')

# for IRNet
parser.add_argument( '--device',       type=str, default='', help='device (cuda:0 or cpu)' )

# for LiNet
parser.add_argument( '--gamma',       type=int, default=0, help='if images need gamma correction 1 = Yes, 0 = No, -1 = invgamma for albedo' )
parser.add_argument( '--batchSize',       type=int, default=8 )

# Image Picking
opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]

nepochs = [opt.nepoch0]


os.system('mkdir {0}'.format(opt.testRoot))

opt.seed = 0
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.arch == 'AlbanNet':
    rootdir = 'AlbanNet/'
    Net = utils_test.AlbanNet(checkpointdir=opt.checkpointdir, cuda=opt.cuda, epoch = opt.nepoch0, nb_channels = opt.nb_channels)

Net.set4testing()

imBatchSmall = Variable(torch.FloatTensor(opt.batchSize, 3, opt.envRow, opt.envCol))


# import pdb; pdb.set_trace()
####################################
outfilename = opt.testRoot + '/results_' + opt.imagedir_out
for n in range(0, opt.level):
    outfilename = outfilename
os.system('mkdir -p {0}'.format(outfilename))

if opt.testset == 'TwoCubesBlendereevee':
    dataRoot = 'testsets/images_eevee'
elif opt.testset == 'TwoCubesBlender128':
    dataRoot = 'testsets/images_128'


with open(os.path.join(dataRoot, opt.imList), 'r') as imIdIn:
    imIds = imIdIn.readlines()
imList = [osp.join(dataRoot, x.strip()) for x in imIds]
imList = sorted(imList)
# import pdb; pdb.set_trace()

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
    j += 1
    print(j)
    print('%d/%d' % (j, nb_iter))
    imOutputNames=[]
    for imName in imNames:
        imId = imName.split('/')[-1]
        print(imId)
        imOutputNames.append(osp.join(outfilename, imId))

    # compiling the batch
    batch = []
    for im_name in imNames:
        batch.append(load_and_process_exr(im_name))
    # Load the image from cpu to gpu


    #import pdb; pdb.set_trace()
    if opt.arch == 'AlbanNet':
        albedoPreds, illuPreds = Net.runtest(batch)

    #################### Output Results #######################

    ## save outputs
    for i, imOutName in enumerate(imOutputNames):
        gt_albedo = load_and_process_exr(imNames[i].replace('.exr','_ref.exr'))
        if opt.nb_channels == 1:
            im_cpu = batch[i].cpu().numpy()
            im_cpu = im_cpu.transpose([1,2, 0])
            gt_albedo = gt_albedo[:,:,0][:, :, np.newaxis]
            gt_albedo2 = gt_albedo.copy()
            gt_albedo2[gt_albedo==0] = 0.00001
            gt_illuminance = im_cpu/gt_albedo2
        #import pdb; pdb.set_trace()

        cv2.imwrite(imOutName[:-4] + '_albedo0.exr', albedoPreds[i].astype("float32"))
        cv2.imwrite(imOutName[:-4] + '.exr', im_cpu.astype("float32"))
        cv2.imwrite(imOutName[:-4] + '_ref.exr', gt_albedo.astype("float32"))
        cv2.imwrite(imOutName[:-4] + '_illuref.exr', gt_illuminance[i:].astype("float32"))
        cv2.imwrite(imOutName[:-4] + '_illu.exr', illuPreds[i].astype("float32"))




