import pdb

import models
import torch, torch.nn as nn
import cv2
from torch.autograd import Variable
import sys, os, time, math, numpy as np

sys.path.append("/LiNet")
import torchvision.utils as vutils
# import YuSmith.lib.irnet as irnet
import YuSmith.lib.torch_util as tu
import torch.optim as optim
# from LiNet import dataLoader
from LiNet import utils
from torch.utils.data import DataLoader
from AlbanNet import dataLoader
from AlbanNet.Unet import UNet
import lib_alban.smoothing as smoothing_lib


def mse_loss(input, target=0):
    return torch.mean((input - target) ** 2)


def compute_smooth_loss(smooth, input):
    scale_factor = 1
    input_weighed = smooth(
        input, patch_size=3, alpha=10, scale_factor=scale_factor
    )
    smoothloss = (
        mse_loss(nn.functional.interpolate(input, scale_factor=scale_factor),
                 input_weighed)
    )
    return smoothloss


class LiNet:
    '''
    Model from Li et al., 2020, adapted and simplified for our purpose.
    It only includes the first cascade and the 4 decoders for intrinsic decomposition.
    '''

    def __init__(self, dataset, experiment, batchsize=2, imWidth0=240, imHeight0=320, cuda=True, gpuID=[0], envRow=120,
                 envCol=160, gamma=False):
        self.rootdir = 'LiNet/'
        self.imWidth = imWidth0
        self.imHeight = imHeight0

        self.experiment = experiment
        self.batchSize = batchsize
        self.cuda = cuda
        self.gpuID = gpuID
        self.imHeights = [imHeight0]
        self.imWidths = [imWidth0]
        self.envRow = envRow
        self.envCol = envCol
        self.gamma = gamma
        self.dataset = dataset

        self.encoder = models.encoder0(cascadeLevel=0)
        self.albedoDecoder = models.decoder0(mode=0)
        self.normalDecoder = models.decoder0(mode=1)
        self.roughDecoder = models.decoder0(mode=2)
        self.depthDecoder = models.decoder0(mode=4)

        if dataset == 'img-geo-small':
            self.dataRoot = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/img-geo-small'
        elif dataset == 'OpenRooms_small':
            self.dataRoot = '/media/alban/T7 Shield/OpenRoomsDataset/SmallOpenRoomsDataset'

        if cuda:
            self.encoder = self.encoder.cuda(self.gpuID)
            self.albedoDecoder = self.albedoDecoder.cuda(self.gpuID)
            self.normalDecoder = self.normalDecoder.cuda(self.gpuID)
            self.roughDecoder = self.roughDecoder.cuda(self.gpuID)
            self.depthDecoder = self.depthDecoder.cuda(self.gpuID)

        lr_scale = 0.5
        self.opEncoder = optim.Adam(self.encoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999))
        self.opAlbedo = optim.Adam(self.albedoDecoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999))
        self.opNormal = optim.Adam(self.normalDecoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999))
        self.opRough = optim.Adam(self.roughDecoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999))
        self.opDepth = optim.Adam(self.depthDecoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999))

    def train(self, epoch, writer):
        albeW, normW = 1.5, 1
        rougW = 0.5
        deptW = 0.5
        trainingLog = open('{0}/trainingLog_{1}.txt'.format(self.experiment, epoch), 'w')

        ####################################
        if 'OpenRooms' in self.dataset:
            brdfDataset = dataLoader.BatchLoader(self.dataRoot,
                                                 imWidth=self.imWidth, imHeight=self.imHeight,
                                                 cascadeLevel=1)
            brdfLoader = DataLoader(brdfDataset, batch_size=self.batchSize,
                                    num_workers=8, shuffle=True)
        elif 'img-geo' in self.dataset:
            brdfLoader = tu.gen(self.dataRoot, colchan=3, batchsize=self.batchsize, tensor=True, device=self.device)

        self.j = 0
        albedoErrsNpList = np.ones([1, 1], dtype=np.float32)
        normalErrsNpList = np.ones([1, 1], dtype=np.float32)
        roughErrsNpList = np.ones([1, 1], dtype=np.float32)
        depthErrsNpList = np.ones([1, 1], dtype=np.float32)

        for i, dataBatch in enumerate(brdfLoader):
            self.j += 1
            # Load data from cpu to gpu
            albedo_cpu = dataBatch['albedo']
            albedoBatch = Variable(albedo_cpu).cuda()

            normal_cpu = dataBatch['normal']
            normalBatch = Variable(normal_cpu).cuda()

            if 'OpenRooms' in self.dataset:
                rough_cpu = dataBatch['rough']
                roughBatch = Variable(rough_cpu).cuda()

                depth_cpu = dataBatch['depth']
                depthBatch = Variable(depth_cpu).cuda()

                segArea_cpu = dataBatch['segArea']
                segEnv_cpu = dataBatch['segEnv']
                segObj_cpu = dataBatch['segObj']

                seg_cpu = torch.cat([segArea_cpu, segEnv_cpu, segObj_cpu], dim=1)
                segBatch = Variable(seg_cpu).cuda()

                segBRDFBatch = segBatch[:, 2:3, :, :]
                segAllBatch = segBatch[:, 0:1, :, :] + segBatch[:, 2:3, :, :]

            # Load the image from cpu to gpu
            im_cpu = (dataBatch['im'])
            imBatch = Variable(im_cpu).cuda()

            # Clear the gradient in optimizer
            self.opEncoder.zero_grad()
            self.opAlbedo.zero_grad()
            self.opNormal.zero_grad()
            self.opRough.zero_grad()
            self.opDepth.zero_grad()

            ########################################################
            # Build the cascade network architecture #
            albedoPreds = []
            normalPreds = []
            roughPreds = []
            depthPreds = []

            inputBatch = imBatch

            # Initial Prediction
            x1, x2, x3, x4, x5, x6 = self.encoder(inputBatch)
            albedoPred = 0.5 * (self.albedoDecoder(imBatch, x1, x2, x3, x4, x5, x6) + 1)
            normalPred = self.normalDecoder(imBatch, x1, x2, x3, x4, x5, x6)
            roughPred = self.roughDecoder(imBatch, x1, x2, x3, x4, x5, x6)
            depthPred = 0.5 * (self.depthDecoder(imBatch, x1, x2, x3, x4, x5, x6) + 1)

            if "OpenRooms" in self.dataset:
                albedoBatch = segBRDFBatch * albedoBatch
            # albedoPred = models.LSregress(albedoPred * segBRDFBatch.expand_as(albedoPred),
            #                              albedoBatch * segBRDFBatch.expand_as(albedoBatch), albedoPred)
            # albedoPred = torch.clamp(albedoPred, 0, 1)

            # depthPred = models.LSregress(depthPred * segAllBatch.expand_as(depthPred),
            #                             depthBatch * segAllBatch.expand_as(depthBatch), depthPred)

            albedoPreds.append(albedoPred)
            normalPreds.append(normalPred)
            roughPreds.append(roughPred)
            depthPreds.append(depthPred)

            ########################################################

            # Compute the error
            albedoErrs = []
            normalErrs = []
            roughErrs = []
            depthErrs = []

            if "OpenRooms" in self.dataset:
                pixelObjNum = (torch.sum(segBRDFBatch).cpu().data).item()
                pixelAllNum = (torch.sum(segAllBatch).cpu().data).item()
            for n in range(0, len(albedoPreds)):
                if "OpenRooms" in self.dataset:
                    albedoErrs.append(torch.sum((albedoPreds[n] - albedoBatch)
                                                * (albedoPreds[n] - albedoBatch) * segBRDFBatch.expand_as(
                        albedoBatch)) / pixelObjNum / 3.0)
                else:
                    albedoErrs.append(torch.sum((albedoPreds[n] - albedoBatch)
                                                * (albedoPreds[n] - albedoBatch)) / albedoBatch.size)
            for n in range(0, len(normalPreds)):
                if "OpenRooms" in self.dataset:
                    normalErrs.append(torch.sum((normalPreds[n] - normalBatch)
                                                * (normalPreds[n] - normalBatch) * segAllBatch.expand_as(
                        normalBatch)) / pixelAllNum / 3.0)
                else:
                    normalErrs.append(torch.sum((normalPreds[n] - normalBatch)
                                                * (normalPreds[n] - normalBatch)) / normalBatch.size)

            if "OpenRooms" in self.dataset:
                for n in range(0, len(roughPreds)):
                    roughErrs.append(torch.sum((roughPreds[n] - roughBatch)
                                               * (roughPreds[n] - roughBatch) * segBRDFBatch) / pixelObjNum)
                for n in range(0, len(depthPreds)):
                    depthErrs.append(torch.sum((torch.log(depthPreds[n] + 1) - torch.log(depthBatch + 1))
                                               * (torch.log(depthPreds[n] + 1) - torch.log(
                        depthBatch + 1)) * segAllBatch.expand_as(depthBatch)) / pixelAllNum)

            # Back propagate the gradients
            if "OpenRooms" in self.dataset:
                totalErr = 4 * albeW * albedoErrs[-1] + normW * normalErrs[-1] \
                           + rougW * roughErrs[-1] + deptW * depthErrs[-1]
            else:
                totalErr = 4 * albeW * albedoErrs[-1] + normW * normalErrs[-1]
            totalErr.backward()
            # import pdb; pdb.set_trace()
            if self.j % 10 == 0:
                writer.add_scalar('Err Albedo', albedoErrs[-1], self.j)
                writer.add_scalar('Err Norm', normalErrs[-1], self.j)
                if "OpenRooms" in self.dataset:
                    writer.add_scalar('Err Rough', roughErrs[-1], self.j)
                    writer.add_scalar('Err Depth', depthErrs[-1], self.j)
                writer.add_scalar('Err Tot', totalErr, self.j)
            if self.j % 50 == 0:
                writer.add_image('Input', vutils.make_grid(imBatch), self.j)
                writer.add_image('Ref Albedos', vutils.make_grid(albedoBatch[0]), self.j)
                writer.add_image('Pred Albedos', vutils.make_grid(albedoPreds[0]), self.j)

            # Update the network parameter
            self.opEncoder.step()
            self.opAlbedo.step()
            self.opNormal.step()
            if "OpenRooms" in self.dataset:
                self.opRough.step()
                self.opDepth.step()

            # Output training error
            utils.writeErrToScreen('albedo', albedoErrs, epoch, self.j)
            utils.writeErrToScreen('normal', normalErrs, epoch, self.j)
            if "OpenRooms" in self.dataset:
                utils.writeErrToScreen('rough', roughErrs, epoch, self.j)
                utils.writeErrToScreen('depth', depthErrs, epoch, self.j)

            utils.writeErrToFile('albedo', albedoErrs, trainingLog, epoch, self.j)
            utils.writeErrToFile('normal', normalErrs, trainingLog, epoch, self.j)
            if "OpenRooms" in self.dataset:
                utils.writeErrToFile('rough', roughErrs, trainingLog, epoch, self.j)
                utils.writeErrToFile('depth', depthErrs, trainingLog, epoch, self.j)

            albedoErrsNpList = np.concatenate([albedoErrsNpList, utils.turnErrorIntoNumpy(albedoErrs)], axis=0)
            normalErrsNpList = np.concatenate([normalErrsNpList, utils.turnErrorIntoNumpy(normalErrs)], axis=0)
            if "OpenRooms" in self.dataset:
                roughErrsNpList = np.concatenate([roughErrsNpList, utils.turnErrorIntoNumpy(roughErrs)], axis=0)
                depthErrsNpList = np.concatenate([depthErrsNpList, utils.turnErrorIntoNumpy(depthErrs)], axis=0)

            if self.j < 1000:
                utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[1:self.j + 1, :], axis=0), epoch,
                                         self.j)
                utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[1:self.j + 1, :], axis=0), epoch,
                                         self.j)
                if "OpenRooms" in self.dataset:
                    utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[1:self.j + 1, :], axis=0), epoch,
                                             self.j)
                    utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[1:self.j + 1, :], axis=0), epoch,
                                             self.j)
                utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[1:self.j + 1, :], axis=0), trainingLog,
                                       epoch,
                                       self.j)
                utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[1:self.j + 1, :], axis=0), trainingLog,
                                       epoch,
                                       self.j)
                if "OpenRooms" in self.dataset:
                    utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[1:self.j + 1, :], axis=0), trainingLog,
                                           epoch, self.j)
                    utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[1:self.j + 1, :], axis=0), trainingLog,
                                           epoch, self.j)

            else:
                utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[self.j - 999:self.j + 1, :], axis=0),
                                         epoch, self.j)
                utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[self.j - 999:self.j + 1, :], axis=0),
                                         epoch, self.j)
                if "OpenRooms" in self.dataset:
                    utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[self.j - 999:self.j + 1, :], axis=0),
                                             epoch, self.j)
                    utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[self.j - 999:self.j + 1, :], axis=0),
                                             epoch, self.j)

                utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[self.j - 999:self.j + 1, :], axis=0),
                                       trainingLog,
                                       epoch, self.j)
                utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[self.j - 999:self.j + 1, :], axis=0),
                                       trainingLog,
                                       epoch, self.j)
                if "OpenRooms" in self.dataset:
                    utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[self.j - 999:self.j + 1, :], axis=0),
                                           trainingLog,
                                           epoch, self.j)
                    utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[self.j - 999:self.j + 1, :], axis=0),
                                           trainingLog,
                                           epoch, self.j)

            '''if self.j == 1 or self.j % 2000 == 0:

                # Save the predicted results
                for n in range(0, len(albedoPreds)):
                    vutils.save_image(((albedoPreds[n]) ** (1.0 / 2.2)).data,
                                      '{0}/{1}_albedoPred_{2}.png'.format(self.experiment, self.j, n))
                for n in range(0, len(normalPreds)):
                    vutils.save_image((0.5 * (normalPreds[n] + 1)).data,
                                      '{0}/{1}_normalPred_{2}.png'.format(self.experiment, self.j, n))
                for n in range(0, len(roughPreds)):
                    vutils.save_image((0.5 * (roughPreds[n] + 1)).data,
                                      '{0}/{1}_roughPred_{2}.png'.format(self.experiment, self.j, n))
                for n in range(0, len(depthPreds)):
                    depthOut = 1 / torch.clamp(depthPreds[n] + 1, 1e-6, 10) * segAllBatch.expand_as(depthPreds[n])
                    vutils.save_image((depthOut * segAllBatch.expand_as(depthPreds[n])).data,
                                      '{0}/{1}_depthPred_{2}.png'.format(self.experiment, self.j, n))'''
        trainingLog.close()

        # Update the training rate
        if (epoch + 1) % 10 == 0:
            for param_group in self.opEncoder.param_groups:
                param_group['lr'] /= 2
            for param_group in self.opAlbedo.param_groups:
                param_group['lr'] /= 2
            for param_group in self.opNormal.param_groups:
                param_group['lr'] /= 2
            for param_group in self.opRough.param_groups:
                param_group['lr'] /= 2
            for param_group in self.opDepth.param_groups:
                param_group['lr'] /= 2
        # Save the error record
        np.save('{0}/albedoError_{1}.npy'.format(self.experiment, epoch), albedoErrsNpList)
        np.save('{0}/normalError_{1}.npy'.format(self.experiment, epoch), normalErrsNpList)
        np.save('{0}/roughError_{1}.npy'.format(self.experiment, epoch), roughErrsNpList)
        np.save('{0}/depthError_{1}.npy'.format(self.experiment, epoch), depthErrsNpList)

        # save the models
        torch.save(self.encoder.module, '{0}/encoder{1}_{2}.pth'.format(self.experiment, 1, epoch))
        torch.save(self.albedoDecoder.module, '{0}/albedo{1}_{2}.pth'.format(self.experiment, 1, epoch))
        torch.save(self.normalDecoder.module, '{0}/normal{1}_{2}.pth'.format(self.experiment, 1, epoch))
        torch.save(self.roughDecoder.module, '{0}/rough{1}_{2}.pth'.format(self.experiment, 1, epoch))
        torch.save(self.depthDecoder.module, '{0}/depth{1}_{2}.pth'.format(self.experiment, 1, epoch))


class IRNet():

    def __init__(self, dataset, batchsize=2, checkpointdir='YuSmith/local-/models/chkpt_test', cuda=True):
        self.rootdir = 'YuSmith/'
        self.device = 'cuda:0' if cuda == True else tu.device()
        self.net = irnet.IRNet_wnorm(3).to(self.device)
        # tu.load(model=self.net, device=self.device, filename= self.rootdir + checkpointdir , require=False)
        # choose loss criterion and optimizer
        self.criterion = nn.MSELoss()  # try others
        self.optimizer = torch.optim.Adam(self.net.parameters(), weight_decay=1e-5)  # check additional parameters
        self.dataset = dataset
        self.batchsize = batchsize
        self.modelfile = checkpointdir + '/chkpt.pt'

        self.iter = 0
        self.starttime = time.time()
        if cuda:
            self.device = 'cuda:0'
        os.system('mkdir {0}'.format(checkpointdir))

        if dataset == 'img-geo-small':
            self.dataRoot = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/img-geo-small/train'
        elif dataset == 'OpenRooms':
            self.dataRoot = '/media/alban/T7 Shield/OpenRoomsDataset/SmallOpenRoomsDataset'

    def train(self, epoch, writer):

        self.net.train()
        vlosslist1 = []

        try:
            epoch_starttime = time.time()
            tlosslist2 = []
            tgen = tu.gen(self.dataRoot, colchan=3, batchsize=self.batchsize, tensor=True, device=self.device)
            for batch, dataBatch in enumerate(tgen):
                lum_in = dataBatch['im']
                albedo_label = dataBatch['albedo']
                normal_label = dataBatch['normal']

                print('%d, %d' % (epoch, batch))
                self.iter += 1

                # forward
                albedo_hat, normal_hat = self.net.forward(lum_in)
                loss = self.criterion(albedo_hat, albedo_label)

                # backward
                self.optimizer.zero_grad()
                loss.backward()

                # adjust weights
                self.optimizer.step()

                # show progress
                tlosslist2.append(loss.item())
                if self.iter % 10 == 0:
                    writer.add_scalar('Err Albedo', loss, self.iter)

                if self.iter % 50 == 0:
                    writer.add_image('Input', vutils.make_grid(lum_in), self.iter)
                    writer.add_image('Ref Albedos', vutils.make_grid(albedo_label), self.iter)
                    writer.add_image('Pred Albedos', vutils.make_grid(albedo_hat), self.iter)
            tu.save(self.net, self.modelfile)

            print(f'epoch elapsed: {self.starttime - epoch_starttime:.2f}\n')
        except:
            tu.save(self.net, self.modelfile)
            raise
        return writer


class AlbanNet_supervised:
    '''
    Used code from Li et al., 2020 as a basis.
    Modified in length for our purpose.
    '''

    def __init__(self, dataset, experiment, batchsize=2, imWidth0=256, imHeight0=256, cuda=True, gpuID=[0],
                 nb_channels=1, smoothing=True):
        self.rootdir = 'LiNet/'
        self.imWidth = imWidth0
        self.imHeight = imHeight0

        self.experiment = 'outs/' + experiment
        self.batchSize = batchsize
        self.cuda = cuda
        self.gpuID = gpuID
        self.imHeights = [imHeight0]
        self.imWidths = [imWidth0]
        self.dataset = dataset
        self.nb_channels = nb_channels

        self.encoder = models.encoder0(cascadeLevel=0, nb_channels=nb_channels)
        self.albedoDecoder = models.decoder0(mode=0, nb_channels=nb_channels)
        self.illuDecoder = models.decoder0(mode=0, nb_channels=nb_channels)
        self.smoothing = smoothing
        if self.nb_channels == 1:
            self.smooth = smoothing_lib.WeightedAverage()
        else:
            self.smooth = smoothing_lib.WeightedAverage() # placeholder
        # self.renderer = UNet(n_channels=9, n_classes=3, bilinear=False)

        self.dataRoot = dataset

        if cuda:
            self.encoder = self.encoder.cuda(self.gpuID)
            self.albedoDecoder = self.albedoDecoder.cuda(self.gpuID)
            # self.normalDecoder = self.normalDecoder.cuda(self.gpuID)
            self.illuDecoder = self.illuDecoder.cuda(self.gpuID)
            # self.renderer = self.renderer.cuda(self.gpuID)

        lr_scale = 0.5
        self.opEncoder = optim.Adam(self.encoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999))
        self.opAlbedo = optim.Adam(self.albedoDecoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999))
        # self.opNormal = optim.Adam(self.normalDecoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999))
        self.opIllu = optim.Adam(self.illuDecoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999))
        # self.opRenderer = optim.Adam(self.renderer.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999))

    def train(self, nbepoch, writer):
        albeW, normW = 150, 1
        illuW = 100
        smoothW = 100
        rendW = 100

        ####################################

        self.j = 0
        albedoErrsNpList = np.ones([1, 1], dtype=np.float32)
        # normalErrsNpList = np.ones([1, 1], dtype=np.float32)
        illuErrsNpList = np.ones([1, 1], dtype=np.float32)
        rendErrsNpList = np.ones([1, 1], dtype=np.float32)

        for epoch in range(nbepoch):
            self.encoder.train()
            self.albedoDecoder.train()
            self.illuDecoder.train()

            brdfDataset = dataLoader.BatchLoader_AlbanNet(self.dataRoot, imWidth=256, imHeight=256, nb_channels=self.nb_channels,
                                                          supervised=True)
            brdfLoader = DataLoader(brdfDataset, batch_size=self.batchSize,
                                    num_workers=8, shuffle=True)
            trainingLog = open('{0}/trainingLog_{1}.txt'.format(self.experiment, epoch), 'w')
            for i, dataBatch in enumerate(brdfLoader):
                self.j += 1
                # Load data from cpu to gpu
                albedo_cpu = dataBatch['albedo']
                albedoBatch = Variable(albedo_cpu).cuda()

                illu_cpu = dataBatch['illu']
                illuBatch = Variable(illu_cpu).cuda()

                # Load the image from cpu to gpu
                im_cpu = (dataBatch['im'])
                imBatch = Variable(im_cpu).cuda()

                # Clear the gradient in optimizer
                self.opEncoder.zero_grad()
                self.opAlbedo.zero_grad()
                self.opIllu.zero_grad()
                # self.opRenderer.zero_grad()

                ########################################################
                # Build the cascade network architecture #
                albedoPreds = []
                # normalPreds = []
                illuPreds = []
                rendererPreds = []

                inputBatch = imBatch

                # Initial Prediction
                x1, x2, x3, x4, x5, x6 = self.encoder(inputBatch)
                albedoPred = 0.5 * (self.albedoDecoder(imBatch, x1, x2, x3, x4, x5, x6) + 1)
                # normalPred = self.normalDecoder(imBatch, x1, x2, x3, x4, x5, x6)
                illuPred = 0.5 * (self.illuDecoder(imBatch, x1, x2, x3, x4, x5, x6) + 1)
                # rendererPred = self.renderer(torch.cat((albedoPred, illuPred), dim = 1))
                rendererPred = albedoPred * illuPred

                ########################################################

                # Compute the error
                albedoErrs = torch.mean(torch.square(albedoPred - albedoBatch))

                illuErrs = torch.mean(torch.square(illuPred - illuBatch))
                # rendErrs = torch.mean(torch.abs(albedoPreds*illuPreds - imBatch)) # when we know the groundtruth reconstruction
                rendErrs = torch.mean(torch.square(rendererPred - imBatch))
                smoothErrs = torch.tensor([0])
                if self.smoothing:
                    smoothErrAlb = compute_smooth_loss(self.smooth, albedoPred)

                    smoothErrIllu = compute_smooth_loss(self.smooth, illuPred)

                    smoothErrRend = compute_smooth_loss(self.smooth, rendererPred)

                    smoothErrs = 2 * smoothErrAlb + 2 * (
                        smoothErrIllu) + 2 * smoothErrRend
                    # Back propagate the gradients
                    totalErr = albeW * albedoErrs + illuW * illuErrs + rendW * rendErrs + smoothW * smoothErrs
                else:
                    totalErr = albeW * albedoErrs + illuW * illuErrs + rendW * rendErrs
                totalErr.backward()
                # import pdb; pdb.set_trace()
                if self.j % 10 == 0:
                    writer.add_scalar('Err Albedo', albedoErrs, self.j)
                    # writer.add_scalar('Err Norm', normalErrs[-1], self.j)
                    writer.add_scalar('Err Illu', illuErrs, self.j)
                    writer.add_scalar('Err Rend', rendW * rendErrs, self.j)
                    writer.add_scalar('Err smooth', smoothW * smoothErrs, self.j)
                    writer.add_scalar('Err Tot', totalErr, self.j)

                if self.j % 400 == 0:
                    # import pdb; pdb.set_trace()
                    writer.add_image('Input', vutils.make_grid((imBatch) ** (1 / 2.2)), self.j)
                    # writer.add_image('Pred Input', vutils.make_grid((albedoPreds[0]*illuPreds[0])**(1/2.2)), self.j)
                    writer.add_image('Input Pred', vutils.make_grid(rendererPred ** (1 / 2.2)), self.j)
                    writer.add_image('Albedos Ref', vutils.make_grid(albedoBatch ** (1 / 2.2)), self.j)
                    writer.add_image('Albedos Pred', vutils.make_grid(albedoPred ** (1 / 2.2)), self.j)
                    # writer.add_image('Ref Normals', vutils.make_grid(normalBatch), self.j)
                    # writer.add_image('Pred Normals', vutils.make_grid(normalPreds[0]), self.j)
                    writer.add_image('Illus Ref', vutils.make_grid(illuBatch ** (1 / 2.2)), self.j)
                    writer.add_image('Illus Pred', vutils.make_grid(illuPred ** (1 / 2.2)), self.j)
                    writer.add_image('Input Rend', vutils.make_grid(illuPred ** (1 / 2.2)), self.j)

                # Update the network parameter
                self.opEncoder.step()
                self.opAlbedo.step()
                # self.opNormal.step()
                self.opIllu.step()
                # self.opRenderer.step()

                # Output training error
                utils.writeErrToScreen('albedo', [albedoErrs], epoch, self.j)
                # utils.writeErrToScreen('normal', normalErrs, epoch, self.j)
                utils.writeErrToScreen('illu', [illuErrs], epoch, self.j)
                utils.writeErrToScreen('rend', [rendErrs], epoch, self.j)
                utils.writeErrToScreen('smooth', [smoothErrs], epoch, self.j)

                utils.writeErrToFile('albedo', [albedoErrs], trainingLog, epoch, self.j)
                # utils.writeErrToFile('normal', normalErrs, trainingLog, epoch, self.j)
                utils.writeErrToFile('illu', [illuErrs], trainingLog, epoch, self.j)
                utils.writeErrToFile('rend', [rendErrs], trainingLog, epoch, self.j)

                albedoErrsNpList = np.concatenate([albedoErrsNpList, utils.turnErrorIntoNumpy([albedoErrs])], axis=0)
                # normalErrsNpList = np.concatenate([normalErrsNpList, utils.turnErrorIntoNumpy(normalErrs)], axis=0)
                illuErrsNpList = np.concatenate([illuErrsNpList, utils.turnErrorIntoNumpy([illuErrs])], axis=0)
                rendErrsNpList = np.concatenate([rendErrsNpList, utils.turnErrorIntoNumpy([rendErrs])], axis=0)
                if self.j % 500 == 0:
                    utils.writeImageToFile(((albedoPred) ** (1.0 / 2.2)).data,
                                           '{0}/{1}_albedopred1.png'.format(self.experiment, self.j))
                    utils.writeImageToFile(((imBatch) ** (1.0 / 2.2)).data,
                                           '{0}/{1}_input1.png'.format(self.experiment, self.j))
                    utils.writeImageToFile(((illuPred[:, :, 10:-10, 10:-10]) ** (1.0 / 2.2)).data,
                                           '{0}/{1}_illupred1.png'.format(self.experiment, self.j))
                    utils.writeImageToFile(((rendererPred) ** (1.0 / 2.2)).data,
                                           '{0}/{1}_rendPred1.png'.format(self.experiment, self.j))
                if self.j < 10000:
                    utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[1:self.j + 1, :], axis=0), epoch,
                                             self.j)
                    # utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[1:self.j + 1, :], axis=0), epoch, self.j)

                    utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[1:self.j + 1, :], axis=0),
                                           trainingLog, epoch,
                                           self.j)
                    utils.writeNpErrToFile('rendAccu', np.mean(rendErrsNpList[1:self.j + 1, :], axis=0), trainingLog,
                                           epoch,
                                           self.j)

                else:
                    utils.writeNpErrToScreen('albedoAccu',
                                             np.mean(albedoErrsNpList[self.j - 999:self.j + 1, :], axis=0), epoch,
                                             self.j)
                    # utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[self.j - 999:self.j + 1, :], axis=0), epoch, self.j)

                    utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[self.j - 999:self.j + 1, :], axis=0),
                                           trainingLog,
                                           epoch, self.j)
                    utils.writeNpErrToFile('rendAccu', np.mean(rendErrsNpList[self.j - 999:self.j + 1, :], axis=0),
                                           trainingLog,
                                           epoch, self.j)

            trainingLog.close()

            # Update the training rate
            if (epoch + 1) % 10 == 0:
                for param_group in self.opEncoder.param_groups:
                    param_group['lr'] /= 2
                for param_group in self.opAlbedo.param_groups:
                    param_group['lr'] /= 2
                for param_group in self.opIllu.param_groups:
                    param_group['lr'] /= 2
                # for param_group in self.opRenderer.param_groups:
                #    param_group['lr'] /= 2
            # Save the error record
            np.save('{0}/albedoError_{1}.npy'.format(self.experiment, epoch), albedoErrsNpList)
            # np.save('{0}/normalError_{1}.npy'.format(self.experiment, epoch), normalErrsNpList)
            np.save('{0}/illuError_{1}.npy'.format(self.experiment, epoch), illuErrsNpList)
            np.save('{0}/rendError_{1}.npy'.format(self.experiment, epoch), rendErrsNpList)

            # save the models
            torch.save(self.encoder, '{0}/encoder{1}_{2}.pth'.format(self.experiment, 1, epoch))
            torch.save(self.albedoDecoder, '{0}/albedo{1}_{2}.pth'.format(self.experiment, 1, epoch))
            # torch.save(self.normalDecoder, '{0}/normal{1}_{2}.pth'.format(self.experiment, 1, epoch))
            torch.save(self.illuDecoder, '{0}/illu{1}_{2}.pth'.format(self.experiment, 1, epoch))
            # torch.save(self.renderer, '{0}/renderer{1}_{2}.pth'.format(self.experiment, 1, epoch))

        return writer


class AlbanNet_unsupervised:
    '''
    Used code from Li et al., 2020 as a basis.
    Modified in length for our purpose.
    '''

    def __init__(self, dataset, experiment, batchsize=1, imWidth0=256, imHeight0=256, cuda=True, gpuID=[0],
                 nb_channels=1, smoothing=True):
        self.rootdir = 'LiNet/'
        self.imWidth = imWidth0
        self.imHeight = imHeight0

        self.experiment = 'outs/' + experiment
        self.batchSize = batchsize
        self.cuda = cuda
        self.gpuID = gpuID
        self.imHeights = [imHeight0]
        self.imWidths = [imWidth0]
        self.dataset = dataset
        self.nb_channels = nb_channels
        self.smoothing = smoothing

        self.encoder = models.encoder0(cascadeLevel=0, nb_channels=nb_channels)
        self.albedoDecoder = models.decoder0(mode=0, nb_channels=nb_channels)
        self.normalDecoder = models.decoder0(mode=1, nb_channels=nb_channels)
        self.illuDecoder = models.decoder0(mode=0, nb_channels=nb_channels)
        if self.nb_channels == 1:
            self.smooth = smoothing_lib.WeightedAverage()
        else:
            self.smooth = smoothing_lib.WeightedAverage()
        # self.renderer = UNet(n_channels=6, n_classes=3, bilinear=False)

        self.dataRoot = dataset

        if cuda:
            self.encoder = self.encoder.cuda(self.gpuID)
            self.albedoDecoder = self.albedoDecoder.cuda(self.gpuID)
            #self.normalDecoder = self.normalDecoder.cuda(self.gpuID)
            self.illuDecoder = self.illuDecoder.cuda(self.gpuID)
            self.smooth = self.smooth.cuda(self.gpuID)
            # self.renderer = self.renderer.cuda(self.gpuID)

        lr_scale = 0.5
        self.opEncoder = optim.Adam(self.encoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999))
        self.opAlbedo = optim.Adam(self.albedoDecoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999))
        #self.opNormal = optim.Adam(self.normalDecoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999))
        self.opIllu = optim.Adam(self.illuDecoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999))
        # self.opRenderer = optim.Adam(self.renderer.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999))

    def train(self, nbepoch, writer, contraW=1000, contraLoss='triplet'):
        triW = contraW
        rendW = 100
        smoothW = 100

        ####################################

        self.j = 0
        rendErrsNpList = np.ones([1, 1], dtype=np.float32)
        tripletErrsNpList = np.ones([1, 1], dtype=np.float32)

        for epoch in range(nbepoch):

            self.encoder.eval()
            self.albedoDecoder.eval()
            self.illuDecoder.eval()
            brdfDataset = dataLoader.BatchLoader_AlbanNet(self.dataRoot, imWidth=256, imHeight=256, nb_channels=self.nb_channels,
                                                          supervised=False, phase = 'TEST')
            brdfLoader = DataLoader(brdfDataset, batch_size=self.batchSize,
                                    num_workers=16, shuffle=True)
            trainingLog = open('{0}/trainingLog_{1}.txt'.format(self.experiment, epoch), 'w')
            for i, dataBatch in enumerate(brdfLoader):
                self.j += 1
                # Load data from cpu to gpu
                im1_cpu = dataBatch['im1']
                im1Batch = Variable(im1_cpu).cuda()

                im2_cpu = dataBatch['im2']
                im2Batch = Variable(im2_cpu).cuda()

                # images will be loaded in batches (n*3*256*256)
                # we want to know which intrinsic outputs should be used for the triplet loss
                # dataloader needs to output which intrinsic component is kept constant as index.
                # Once we have that, we can stack the intrinsic componnets together, and compute the using the indexes
                non_constant_idx = dataBatch['non_constant_idx']
                constant_idxs = dataBatch['constant_idxs']

                # Clear the gradient in optimizer
                self.opEncoder.zero_grad()
                self.opAlbedo.zero_grad()
                # self.opNormal.zero_grad()
                self.opIllu.zero_grad()
                # self.opRenderer.zero_grad()

                ########################################################
                # Build the cascade network architecture #

                # Initial Prediction 1
                x11, x21, x31, x41, x51, x61 = self.encoder(im1Batch)
                albedoPred1 = 0.5 * (self.albedoDecoder(im1Batch, x11, x21, x31, x41, x51, x61) + 1)

                # normalPred1 = self.normalDecoder(im1Batch, x11, x21, x31, x41, x51, x61)

                illuPred1 = 0.5 * (self.illuDecoder(im1Batch, x11, x21, x31, x41, x51, x61) + 1)

                # rendererPred1 = self.renderer(torch.cat((albedoPred1, illuPred1), dim = 1))
                rendererPred1 = albedoPred1 * illuPred1

                # Initial Prediction 2
                x12, x22, x32, x42, x52, x62 = self.encoder(im2Batch)
                albedoPred2 = 0.5 * (self.albedoDecoder(im2Batch, x12, x22, x32, x42, x52, x62) + 1)

                # normalPred2 = self.normalDecoder(im2Batch, x12, x22, x32, x42, x52, x62)

                illuPred2 = 0.5 * (self.illuDecoder(im2Batch, x12, x22, x32, x42, x52, x62) + 1)

                # rendererPred2 = self.renderer(torch.cat((albedoPred2, illuPred2), dim=1))
                rendererPred2 = albedoPred2 * illuPred2

                # intrinsicPreds1 = torch.stack(( albedoPred1, illuPred1, normalPred1), dim = 1)
                intrinsicPreds1 = torch.stack((albedoPred1, illuPred1), dim=1)  # ignore normals for now
                # intrinsicPreds2= torch.stack((albedoPred2, illuPred2, normalPred2), dim=1)
                intrinsicPreds2 = torch.stack((albedoPred2, illuPred2), dim=1)  # ignore normals for now

                # rendererPreds1 = rendererPred1
                # rendererPreds2 = rendererPred2

                ########################################################
                ### COMPUTE THE LOSS

                # Initiialize
                tripletErrs = torch.zeros(len(im1Batch)).cuda(self.gpuID)
                # diff_intrinisic = torch.mean(torch.square(intrinsicPreds1 - intrinsicPreds2), axis = (2,3,4))
                diff_intrinisic = torch.mean(torch.square(intrinsicPreds1 - intrinsicPreds2), axis=(2, 3, 4))

                ## Contrastive loss
                for m in range(len(tripletErrs)):
                    if contraLoss == 'triplet':
                        tripletErrs[m] = torch.maximum(
                            diff_intrinisic[m, constant_idxs[m]] * 10 - diff_intrinisic[m, non_constant_idx[m]],
                            torch.tensor(0))
                    elif contraLoss == 'constant':
                        tripletErrs[m] = diff_intrinisic[m, constant_idxs[m]]
                tripletErrs = torch.mean(tripletErrs, dim=0, keepdim=True)

                ## Rendering error
                # import pdb; pdb.set_trace()
                rendErrs = 0.5 * torch.mean(torch.square(rendererPred1 - (im1Batch))) + 0.5 * torch.mean(
                    torch.square(rendererPred2 - (im2Batch)))

                # Smoothness losses
                smoothErrs = torch.tensor([0])
                if self.smoothing:
                    smoothErrAlb1 = compute_smooth_loss(self.smooth, albedoPred1)
                    smoothErrAlb2 = compute_smooth_loss(self.smooth, albedoPred2)

                    smoothErrIllu1 = compute_smooth_loss(self.smooth, illuPred1)
                    smoothErrIllu2 = compute_smooth_loss(self.smooth, illuPred2)

                    smoothErrRend1 = compute_smooth_loss(self.smooth, rendererPred1)
                    smoothErrRend2 = compute_smooth_loss(self.smooth, rendererPred2)

                    smoothErrs = 2 * smoothErrAlb1 + 2 * smoothErrAlb2 + 2 * (
                                smoothErrIllu1 + smoothErrIllu2) + 2 * smoothErrRend1 + 2 * smoothErrRend2
                    # Back propagate the gradients

                    totalErr = 4 * triW * tripletErrs + rendW * rendErrs + smoothW * smoothErrs
                else:
                    totalErr = 4 * triW * tripletErrs + rendW * rendErrs
                totalErr.backward()


            np.save('{0}/tripletError_{1}.npy'.format(self.experiment, epoch), tripletErrsNpList)
            np.save('{0}/rendError_{1}.npy'.format(self.experiment, epoch), rendErrsNpList)

            # save the models
            torch.save(self.encoder, '{0}/encoder{1}_{2}.pth'.format(self.experiment, 1, epoch))
            torch.save(self.albedoDecoder, '{0}/albedo{1}_{2}.pth'.format(self.experiment, 1, epoch))
            torch.save(self.normalDecoder, '{0}/normal{1}_{2}.pth'.format(self.experiment, 1, epoch))
            torch.save(self.illuDecoder, '{0}/illu{1}_{2}.pth'.format(self.experiment, 1, epoch))
            # torch.save(self.renderer, '{0}/renderer{1}_{2}.pth'.format(self.experiment, 1, epoch))


        return writer


def centerNnormalize(img):
    img = torch.permute(img, (2, 3, 0, 1))
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min)
    # img = torch.permute(img, (2,3,0,1))
    return img