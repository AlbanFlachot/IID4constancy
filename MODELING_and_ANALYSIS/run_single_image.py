import os
import cv2

import sys

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
parser.add_argument('--image',  type=str, help='path to real image')
# Model
parser.add_argument('--arch', default='AlbanNet', help='the architecture')
parser.add_argument('--checkpointdir', default=None, help='the path to the checkpoint')
# Paths
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

#### INITIALIZATION
opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]

nepochs = [opt.nepoch0]


opt.seed = 0
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)

opt.batchSize = 1
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if 'AlbanNet' in opt.arch:
    rootdir = 'AlbanNet/'
    Net = utils_test.AlbanNet(checkpointdir=opt.checkpointdir, cuda=opt.cuda, epoch = opt.nepoch0, nb_channels = opt.nb_channels)

Net.set4testing()


#### IMAGE PICKING

splitpath = opt.image.split('/')
dataRoot = opt.image[:-len(splitpath[-1])]
imagename = splitpath[-1]
### directory where to save 10 first outputs and inputs
os.system(f'mkdir {dataRoot}outputs')

image = load_and_process_exr(opt.image)


# inference
if 'AlbanNet' in opt.arch:
    albedoPreds, illuPreds = Net.runtest([image])
    #import pdb; pdb.set_trace()
    cv2.imwrite(dataRoot + f'outputs/{imagename[:-4]}' + '_albedo0.exr', albedoPreds[0].astype("float32"))



