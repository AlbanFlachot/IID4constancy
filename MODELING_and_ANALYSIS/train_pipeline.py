import sys

sys.path.append("/LiNet")
sys.path.append("/YuSmith")
import torch

import argparse
import random
import os
import os.path as osp
import lib_alban.utils_train as utils_train
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
# The locationi of testing set
parser.add_argument('--trainset', help='path to real images')
parser.add_argument('--imList', help='path to image list')

# Model
parser.add_argument('--arch', default='IRNet', help='the architecture')
parser.add_argument('--checkpointdir', default=None, help='the path to the checkpoint')

# Paths
parser.add_argument('--trainRoot', help='the path to save the testing errors')
parser.add_argument('--experiment', default='test', help='name of experiment')
parser.add_argument('--imagedir_out', type=str, default='../test_out/')

# The basic testing setting
parser.add_argument('--nbepochs', type=int, default=14, help='the number of epoch for testing')

parser.add_argument('--imHeight0', type=int, default=240, help='the height / width of the input image to network')
parser.add_argument('--imWidth0', type=int, default=320, help='the height / width of the input image to network')
parser.add_argument('--imHeight1', type=int, default=240, help='the height / width of the input image to network')
parser.add_argument('--imWidth1', type=int, default=320, help='the height / width of the input image to network')

parser.add_argument('--envRow', type=int, default=120, help='the height /width of the envmap predictions')
parser.add_argument('--envCol', type=int, default=160, help='the height /width of the envmap predictions')
parser.add_argument('--envHeight', type=int, default=8, help='the height /width of the envmap predictions')
parser.add_argument('--envWidth', type=int, default=16, help='the height /width of the envmap predictions')

parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for testing network')

parser.add_argument('--level', type=int, default=2, help='the cascade level')

parser.add_argument('--nb_channels', type=int, default='1', help='grayscale or RGB')
parser.add_argument('--contraLoss', type=str, default='triplet', help='weight for contra loss')
parser.add_argument('--contraW', type=int, default=1000, help='weight for contra loss')
parser.add_argument('--lrStep', type=int, default=10, help='step for lr decay')

# for IRNet
parser.add_argument('--device', type=str, default='', help='device (cuda:0 or cpu)')
parser.add_argument('--smoothing', action='store_true', default=False, help='add smooth losss')
parser.add_argument('--noise', action='store_true', default=False, help='add smooth losss')

parser.add_argument('--noRend', action='store_true', default=False, help='do not render')

# Image Picking
opt = parser.parse_args()
print(opt)

writer = SummaryWriter(osp.join('outs/runs4VSS', opt.experiment))

opt.gpuId = opt.deviceIds[0]

imHeights = [opt.imHeight0]
imWidths = [opt.imWidth0]

os.system('mkdir {0}'.format(opt.trainRoot))

isExist = os.path.exists('outs/' + opt.experiment)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs('outs/' + opt.experiment)

opt.seed = 0
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)

'''logfile = 'log.txt'
if os.path.exists(logfile):
    os.remove(logfile)'''

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.arch == 'LiNet':
    rootdir = 'LiNet/'
    Net = utils_train.LiNet(dataset=opt.trainset, experiment=opt.experiment, imWidth0=opt.imWidth0,
                            imHeight0=opt.imWidth0,
                            cuda=opt.cuda, gpuID=opt.gpuId, envRow=opt.envRow, envCol=opt.envCol)
elif opt.arch == 'IRNet':
    rootdir = 'YuSmith/'
    Net = utils_train.IRNet(dataset=opt.trainset, cuda=opt.cuda)

elif opt.arch == 'AlbanNet_supervised':
    rootdir = 'LiNet/'
    Net = utils_train.AlbanNet_supervised(dataset=opt.trainset, batchsize=4, experiment=opt.experiment, imWidth0=256,
                                          imHeight0=256, cuda=opt.cuda, gpuID=opt.gpuId, nb_channels=opt.nb_channels,
                                          smoothing=opt.smoothing, lrStep=opt.lrStep, noRend=opt.noRend,
                                          noise=opt.noise)
elif opt.arch == 'AlbanNet_unsupervised':
    rootdir = 'LiNet/'
    # import pdb; pdb.set_trace()
    Net = utils_train.AlbanNet_unsupervised(dataset=opt.trainset, batchsize=2, experiment=opt.experiment, imWidth0=256,
                                            imHeight0=256,
                                            cuda=opt.cuda, gpuID=opt.gpuId, nb_channels=opt.nb_channels,
                                            smoothing=opt.smoothing, lrStep=opt.lrStep)
if opt.arch == 'AlbanNet_unsupervised':
    writer = Net.train(opt.nbepochs, writer, contraW=opt.contraW, contraLoss=opt.contraLoss)
else:
    writer = Net.train(opt.nbepochs, writer)
writer.close()