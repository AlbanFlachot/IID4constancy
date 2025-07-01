import models
import torch
import cv2
import numpy as np
#import YuSmith.lib.irnet as irnet
#import YuSmith.lib.torch_util as tu
import os
from lib_alban.utils_analyse import load_and_process, load_and_process_exr, load_and_process_illusions

def gamma_corr(im):
    im = im/255
    im = im**(1/2)
    im = (im * 255).astype(np.uint8)
    return im

def inv_gamma(im):
    im = im/255
    im = im**2.2
    im = (im*255).astype(np.uint8)
    return im

class AlbanNet:
    '''
    Model from Li et al., 2020, adapted and simplified for our purpose.
    It only includes the first cascade and the 4 decoders for intrinsic decomposition.
    '''

    def __init__(self, checkpointdir='check_cascade0_w%d_h%d', epoch=35, imWidth0=256, imHeight0=256, cuda=True, gpuID=0,
                 envRow=256, envCol=256, gamma=False, nb_channels = 1):
        self.rootdir = 'AlbanNet/'
        if checkpointdir is None:
            self.xp0 = self.rootdir + 'check_cascade0_w%d_h%d' % (imWidth0, imHeight0)
        else:
            self.xp0 = checkpointdir
            self.imWidth = imWidth0
            self.imHeight0 = imHeight0

        self.cuda = cuda
        self.gpuID = gpuID
        self.experiments = [checkpointdir]
        self.nepochs = [epoch]
        self.imHeights = [imHeight0]
        self.imWidths = [imWidth0]
        self.envRow = envRow
        self.envCol = envCol
        self.gamma = gamma
        self.nb_channels = nb_channels

        self.encoders = models.encoder0(cascadeLevel=0, nb_channels = nb_channels)
        self.albedoDecoders = models.decoder0(mode=0, nb_channels = nb_channels)
        self.illuDecoders = models.decoder0(mode=0, nb_channels = nb_channels)


    def set4testing(self):

        self.batchSize = 1

        self.encoders = self.encoders.eval()
        self.albedoDecoders = self.albedoDecoders.eval()
        self.illuDecoders = self.illuDecoders.eval()
        # import pdb; pdb.set_trace()

        # Load weight
        print(os.getcwd() + self.rootdir + '{0}/encoder{1}_{2}.pth'.format(self.xp0, 1, self.nepochs[0] - 1))
        self.encoders.load_state_dict(
            torch.load(self.rootdir + '{0}/encoder{1}_{2}.pth'.format(self.xp0, 1, self.nepochs[0] - 1),  weights_only=False).state_dict())
        self.albedoDecoders.load_state_dict(
            torch.load(self.rootdir + '{0}/albedo{1}_{2}.pth'.format(self.xp0, 1, self.nepochs[0] - 1) , weights_only=False).state_dict())
        self.illuDecoders.load_state_dict(
            torch.load(self.rootdir + '{0}/illu{1}_{2}.pth'.format(self.xp0, 1, self.nepochs[0] - 1), weights_only=False).state_dict())

        for param in self.encoders.parameters():
            param.requires_grad = False
        for param in self.albedoDecoders.parameters():
            param.requires_grad = False
        for param in self.illuDecoders.parameters():
            param.requires_grad = False

        if self.cuda:
            self.encoders = self.encoders.cuda(self.gpuID)
            self.albedoDecoders = self.albedoDecoders.cuda(self.gpuID)
            self.illuDecoders = self.illuDecoders.cuda(self.gpuID)



    def runtest(self, imBatch):
        for i, im_cpu in enumerate(imBatch):
            # Resize Input Images
            if self.nb_channels == 1:
                imBatch[i] = imBatch[i][:,:,0][:, :, np.newaxis]
            #import pdb; pdb.set_trace()
            imBatch[i] = (np.transpose(imBatch[i], [2, 0, 1]).astype(np.float32))[:, :, :]
            imBatch[i] = torch.autograd.Variable(torch.from_numpy(imBatch[i])).cuda()
            # im = im**(1/2.2)
            # im = im / im.max()
            #imBatches.append(torch.autograd.Variable(torch.from_numpy(np.clip(im, 0,1))).cuda())


        ################# BRDF Prediction ######################
        inputBatch = torch.stack(imBatch)

        x1, x2, x3, x4, x5, x6 = self.encoders(inputBatch)

        albedoPred = 0.5 * (self.albedoDecoders(inputBatch, x1, x2, x3, x4, x5, x6) + 1)
        illuPred = self.illuDecoders(inputBatch, x1, x2, x3, x4, x5, x6) + 1
        rendPred = albedoPred*illuPred


        albedoPred = albedoPred.data.cpu().numpy()
        albedoPred = albedoPred.transpose([0,2, 3, 1])


        illuPred = illuPred.data.cpu().numpy()
        illuPred = illuPred.transpose([0, 2, 3, 1])

        return albedoPred, illuPred