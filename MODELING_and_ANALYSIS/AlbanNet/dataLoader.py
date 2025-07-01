import glob
import torch
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import os.path as osp
from PIL import Image
from torch.autograd import Variable
import random
import struct
from torch.utils.data import Dataset
import cv2
import json
from scipy.spatial.transform import Rotation as Rot
from skimage.measure import block_reduce
import h5py
import scipy.ndimage as ndimage


class BatchLoader_AlbanNet(Dataset):
    def __init__(self, dataRoot, dirs=['train'],
                 imHeight=256, imWidth=256,
                 phase='TRAIN', rseed=None, nb_channels=1,
                 envHeight=8, envWidth=16, envRow=120, envCol=160,
                 SGNum=12, supervised=True):

        if phase.upper() == 'TRAIN':
            if supervised:
                self.sceneFile = osp.join(dataRoot, 'train_files.txt')  # train_scenes_expanded.txt
            else:
                self.sceneFile = osp.join(dataRoot, 'train_scenes.txt')  # train_scenes_expanded.txt
        elif phase.upper() == 'TEST':
            if supervised:
                self.sceneFile = osp.join(dataRoot, 'test_files.txt')  # train_scenes_expanded.txt
            else:
                self.sceneFile = osp.join(dataRoot, 'test_scenes.txt')  # train_scenes_expanded.txt
        else:
            print('Unrecognized phase for data loader')
            assert (False)

        with open(self.sceneFile, 'r') as fIn:
            imList = fIn.readlines()

        self.instrinsic_modifs = np.genfromtxt(osp.join(dataRoot, 'intrinsic_modifs.txt'), delimiter=',', dtype=str)
        self.supervised = supervised
        self.imList = [osp.join(dataRoot, x.strip()) for x in imList]  # remove the \n at the end
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.phase = phase.upper()
        self.envWidth = envWidth
        self.envHeight = envHeight
        self.envRow = envRow
        self.envCol = envCol
        self.SGNum = SGNum
        self.nb_channels = nb_channels

        # Permute the image list
        self.count = len(self.imList)
        if self.supervised:
            self.perm = list(range(self.count))  # stack list of idx to ensure epoch with as many images as supervised
        else:
            self.perm = list(range(self.count)) * ((self.instrinsic_modifs[:-1,
                                                    1:].size // 2 + 1))  # stack list of idx to ensure epoch with as many images as supervised
        # if rseed is not None:
        #   random.seed(0)
        np.random.shuffle(self.perm)

        print('Images Num: %d' % (len(self.perm)))
        # print(self.perm[:10])
        self.albedoList = [x.replace('.exr', '_ref.exr') for x in self.imList]
        print(self.albedoList[self.perm[0]])

    def __len__(self):
        return len(self.perm)

    def __getitem__(self, ind):
        # Read segmentation
        # print(ind)
        # randomize intrinsic modification
        curr_scene = self.imList[self.perm[ind]].split('/')[-1].split('_')[0]
        intrinsic_idexes = np.arange(0, self.instrinsic_modifs.shape[0] - 1).astype(int)
        if self.supervised:
            im = self.loadHdr(self.imList[self.perm[ind]])
            #im = im / im.max()  # normalization to fix image to be between 0 and 1
            # Read albedo
            albedo = self.loadHdr(self.albedoList[self.perm[ind]])
            # albedo = (0.5 * (albedo + 1) ) ** 2.2

            if self.nb_channels == 1:  # Case when we have achromatic inputs
                c = np.random.choice(np.arange(0, 3).astype(int), 1)[0]  # randomnly select channel
                im = im[c][np.newaxis, :]  # apply to all inputs
                albedo = albedo[c][np.newaxis, :]
            else:  # Case when we include color channels
                isgray = np.random.random(1)
                if isgray < 0.2:  # 20% chance of showing grayscale
                    c = np.random.choice(np.arange(0, 3).astype(int), 1)[0]  # randomnly select channel
                    im = im[c]  # apply to all inputs
                    im = np.stack((im, im, im), 0)  # stack to have a 3 channel input
                    albedo = albedo[c]
                    albedo = np.stack((albedo, albedo, albedo), 0)  # stack to have a 3 channel input
                # swap channels for augmentation
                idxes = np.array([0, 1, 2])
                np.random.shuffle(idxes)
                im = im[idxes]
                albedo = albedo[idxes]

            im = torch.tensor(im)
            albedo = torch.tensor(albedo)

            # Random flips for augmentation
            flip_x = np.random.choice(np.array([False, True]))
            flip_y = np.random.choice(np.array([False, True]))
            if flip_x:
                im = torch.flip(im, [1])
                albedo = torch.flip(albedo, [1])
            if flip_y:
                im = torch.flip(im, [2])
                albedo = torch.flip(albedo, [2])
            im, illu = self.compute_illuminance(im, albedo)
            #noisy_im = self.addnoise(im)

            batchDict = {'albedo': albedo,
                         'illu': illu,
                         'im': im,
                         'noisy_im': im,
                         'name': self.imList[self.perm[ind]]
                         }


        else:

            non_constant_idx = np.random.choice(intrinsic_idexes, 1)[
                0]  # intrinsic modif type to consider ie albedo, normals or light
            constant_idxs = intrinsic_idexes[intrinsic_idexes != non_constant_idx][
                0]  # intrinsic components staying constant

            """FIX ME: MAKE PROB EQUAL FOR ALL IMAGES"""
            img1_modif = np.random.choice(self.instrinsic_modifs[non_constant_idx, 1:], 1)[
                0]  # for the modif type, which pair to choose
            remaining_choices = self.instrinsic_modifs[non_constant_idx, 1:][self.instrinsic_modifs[non_constant_idx,
                                                                             1:] != img1_modif]  # making sure the 2nd input is not the first one
            img2_modif = np.random.choice(remaining_choices, 1)[0]

            path2imdir = self.imList[self.perm[ind]][:-len(self.imList[self.perm[ind]].split('/')[-1])]
            im1_path = path2imdir + curr_scene + img1_modif
            im2_path = path2imdir + curr_scene + img2_modif
            # print([im1_path, im2_path])  # checking everything is fine

            # Read Image
            im1 = self.loadHdr(im1_path)
            im2 = self.loadHdr(im2_path)
            # Convert to greyscale if needed
            if self.nb_channels == 1:
                c = np.random.choice(np.arange(0, 3).astype(int), 1)[0]  # randomnly select channel
                im1 = im1[c][np.newaxis, :]  # apply to all inputs
                im2 = im2[c][np.newaxis, :]
            else:
                isgray = np.random.random(1)
                if isgray < 0.25:  # 25% chance of showing grayscale
                    c = np.random.choice(np.arange(0, 3).astype(int), 1)[0]  # randomnly select channel
                    im1 = im1[c]  # apply to all inputs
                    im1 = np.stack((im1, im1, im1), 0)  # stack to have a 3 channel input
                    im2 = im2[c]  # apply to all inputs
                    im2 = np.stack((im2, im2, im2), 0)  # stack to have a 3 channel input
                # Random channel swap for augmentation
                idxes = np.array([0, 1, 2])
                np.random.shuffle(idxes)
                im1 = im1[idxes]
                im2 = im2[idxes]
                im1 = torch.tensor(im1)
                im2 = torch.tensor(im2)
            # Random flips for augmentation
            flip_x = np.random.choice(np.array([False, True]))
            flip_y = np.random.choice(np.array([False, True]))
            if flip_x:
                im1 = torch.flip(im1, [1])
                im2 = torch.flip(im2, [1])
            if flip_y:
                im1 = torch.flip(im1, [2])
                im2 = torch.flip(im2, [2])
            noisy_im1 = self.addnoise(im1)
            noisy_im2 = self.addnoise(im2)
            # Read albedo
            # albedo = self.loadHdr(self.albedoList[self.perm[ind]])
            # albedo = (0.5 * (albedo + 1) ) ** 2.2

            # normalize the normal vector so that it will be unit length
            # normal = self.loadHdr(self.normalList[self.perm[ind]])
            # normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5) )[np.newaxis, :]

            # illu = self.compute_illuminance(self.imList[self.perm[ind]], im, normal, albedo)

            batchDict = {'im1': im1,
                         'im2': im2,
                         'noisy_im1': noisy_im1,
                         'noisy_im2': noisy_im2,
                         'constant_idxs': constant_idxs,
                         'non_constant_idx': non_constant_idx,
                         'name': self.imList[self.perm[ind]],
                         'modifs': [img1_modif, img2_modif],
                         'isgray': isgray < 0.25
                         }

        return batchDict

    def loadImage(self, imName, isGama=False):
        if not (osp.isfile(imName)):
            print(imName)
            assert (False)

        im = Image.open(imName)
        im = im.resize([self.imWidth, self.imHeight], Image.ANTIALIAS)

        im = np.asarray(im, dtype=np.float32)
        if isGama:
            im = (im / 255.0) ** 2.2
            im = 2 * im - 1
        else:
            im = (im - 127.5) / 127.5
        if len(im.shape) == 2:
            im = im[:, np.newaxis]
        im = np.transpose(im, [2, 0, 1])

        return im

    def loadHdr(self, imName):
        if not (osp.isfile(imName)):
            print(imName)
            assert (False)
        im = cv2.imread(imName, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if im is None:
            print(imName)
            assert (False)
        # im = cv2.resize(im, (self.imWidth, self.imHeight), interpolation = cv2.INTER_AREA )
        im = np.transpose(im, [2, 0, 1])
        # im = im[::-1, :, :]
        return im

    def compute_illuminance(self, img, albedo):
        ### account for excentricity
        ## need to build a vector image, where each pixel is the direction of camera corrected by excentrixity
        ## i.e. for each pixel, change the camera direction according to excentricity.
        ## view angle directly proportional to pixel position
        ## at center of image, view angle is 0, in extremities view angle at max.
        ## then take dot product at each pixel between normals and vector image
        '''f = open(imName.replace('.exr', '.json'))
        data = json.load(f)
        camera_data = data['camera']
        print(camera_data)
        az = camera_data['azimuth'] * np.pi / 180
        el = camera_data['elevation'] * np.pi / 180

        dir_camera = np.array([(np.cos(az) * np.cos(el)), (np.sin(az) * np.cos(el)), np.sin(el)])


        imgshape = np.array(img.shape[:2])
        FOV = np.array([39.6, 27.0]) * np.pi / 180
        FOV_xpix = np.arange(-FOV[0] / 2, FOV[0] / 2, FOV[0] / imgshape[0])
        FOV_ypix = np.arange(-FOV[1] / 2, FOV[1] / 2, FOV[1] / imgshape[1])
        # img_center = np.array([img.shape[0]/2,img.shape[1]/2])
        X, Y = np.meshgrid(FOV_xpix, FOV_ypix)

        vect_image = np.ones(img.shape) * dir_camera
        for j in range(imgshape[1]):
            rot1 = Rot.from_euler('z', -X[0, j])
            vect_image[:, j] = rot1.apply(vect_image[:, j])
        for i in range(imgshape[0]):
            rot2 = Rot.from_euler('x', -Y[i, j])
            vect_image[i, :] = rot2.apply(vect_image[i, :])'''
        albedo[
            albedo == 0] = 0.000001  # change 0 values to non zero for following division. pixels were albedo = 0 are also 0 in img so result should be the same.
        illuminance = img / (albedo)
        tresh = illuminance.max()/2 # We want to limit illuminance to be at 2 max...
        if tresh >1:
            illuminance = illuminance/tresh
            img = img/tresh
        return img, illuminance

    def addnoise(self, img):
        noisetype = np.random.random(1)
        if noisetype < 0.5:
            noise_mask = torch.randn(tuple(img.shape[1:])) * 0.5
        else:
            noise_mask = torch.rand(tuple(img.shape[1:])) * 2 - 1
        if img.shape[0] == 1:
            noise_mask = torch.unsqueeze(noise_mask, 0)
        if img.shape[0] == 3:
            noise_mask = torch.stack((noise_mask, noise_mask, noise_mask), 0)
        gain = 1 - np.random.power(2, 1)[0]
        img += gain * noise_mask * img
        img = torch.clamp(img, 0, 1)
        return img