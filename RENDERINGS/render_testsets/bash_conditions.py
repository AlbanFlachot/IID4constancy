import subprocess
import shlex

import numpy as np

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import shutil
from os.path import join


intensities = np.round(np.array([0, 0.35, 0.75, 1.5, 3]), 2)
ref_reflectances = np.round(np.array([0.2, 0.4, 0.6]), 1)
test_reflectances = np.array([0.00, 0.025,0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.715, 0.775, 0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975])
#test_reflectances = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])


### Render the masks to be used for the ambiguous conditions
command = 'blender --background --python render_testset_mask.py --python-use-system-env ' \
          '-- --filename testset/masks/rightsheetsmallcycles --renderer CYCLES --samples 128'
args = shlex.split(command)
subprocess.call(args)

'''
command = 'blender --background --python render_patterns_cube.py --python-use-system-env ' \
          '-- --filename testset/testpatterns --renderer EEVEE --samples 128'
args = shlex.split(command)
subprocess.call(args)'''

command = 'blender --background --python render_testset_empty.py --python-use-system-env ' \
          '-- --filename testset/empty_1282p9 --renderer CYCLES --samples 128'
args = shlex.split(command)
subprocess.call(args)


def load_mask(path):
    tresh= 0.99
    mask = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).mean(-1)
    if path[-3:]=='png':
        mask = mask/mask.max()
        mask[mask>tresh]=1.0
    return mask


samples = '128' # or eevee
path2outs = "testsets/"
#dataset = f'images_{samples}_patterns'
dataset = f'images_{samples}2p9'
datasetempty = f'empty_{samples}2p9'


### floor condition
#mask_floor = load_mask(path2outs + 'masks_nov2_patterns/floor_ref.exr')
#mask_floor3D = np.stack((mask_floor, mask_floor, mask_floor), -1)
mask_negativefloor = cv2.imread(path2outs + f'masks/floor_ref.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
mask_floor3D = cv2.cvtColor(mask_negativefloor , cv2.COLOR_BGR2RGB)
mask_floor3D[mask_floor3D>1]=1


for i, illu in enumerate(intensities):
    for r in ref_reflectances:
        for t in test_reflectances:
            imgpath = join(path2outs, "{0}/img_{1}_{2}_{3}.exr".format(dataset, r, t, illu))
            #imgpath_0 = join(path2outs, "{0}/img_{1}_{2}_{3}.exr".format(dataset, r, t, 0.0))
            imgpath_0 = join(path2outs, "{0}/img_{1}.exr".format(datasetempty, illu))
            img = cv2.imread(imgpath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            img_0 = cv2.imread(imgpath_0, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            img_amb_shadows = img_0 * (mask_floor3D) + img * (1 - mask_floor3D)

            cv2.imwrite(path2outs+"{0}/img_{1}_{2}_{3}_floor.exr".format(dataset, r, t, illu), img_amb_shadows.astype(np.float32))
            shutil.copy(path2outs+"{0}/img_{1}_{2}_{3}_ref.exr".format(dataset, r, t, illu), path2outs+"{0}/img_{1}_{2}_{3}_floor_ref.exr".format(dataset, r, t, illu))

### objects condition

# masks of all different objects
#mask_allobj = load_mask(path2outs + 'masks_nov2_patterns/allobjects_ref.exr')
#mask_allobj3D = np.stack((mask_allobj, mask_allobj, mask_allobj), -1)
mask_allobj3D = cv2.imread(path2outs + f'masks/allobj_ref.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
mask_allobj3D = cv2.cvtColor(mask_allobj3D , cv2.COLOR_BGR2RGB)

for i, illu in enumerate(intensities):
    for r in ref_reflectances:
        for t in test_reflectances:
            imgpath = join(path2outs, "{0}/img_{1}_{2}_{3}.exr".format(dataset, r, t, illu))
            imgpath_0 = join(path2outs, "{0}/img_{1}_{2}_{3}.exr".format(dataset, r, t, 0.0))
            img = cv2.imread(imgpath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            img_0 = cv2.imread(imgpath_0, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            # Get average luminance
            meanlum0 = img_0[mask_allobj3D == 1].mean()
            meanlum = img[mask_allobj3D == 1].mean()
            coefflum = meanlum / meanlum0

            img_amb_shading = img_0*(mask_allobj3D)*coefflum + img * (1-mask_allobj3D)

            cv2.imwrite(path2outs+"{0}/img_{1}_{2}_{3}_sphere.exr".format(dataset, r, t, illu), img_amb_shading.astype(np.float32))
            shutil.copy(path2outs+"{0}/img_{1}_{2}_{3}_ref.exr".format(dataset, r, t, illu), path2outs+"{0}/img_{1}_{2}_{3}_sphere_ref.exr".format(dataset, r, t, illu))

### floorsphere condition

# masks of all different objects
mask_objfloor3D = mask_floor3D + mask_allobj3D
mask_objfloor3D[mask_objfloor3D>1] = 1

# Get average luminance
meanlum0 = img_0[mask_allobj3D==1].mean()
meanlum = img[mask_allobj3D==1].mean()
coefflum = meanlum/meanlum0

for i, illu in enumerate(intensities):
    for r in ref_reflectances:
        for t in test_reflectances:
            imgpath = join(path2outs, "{0}/img_{1}_{2}_{3}.exr".format(dataset, r, t, illu))
            imgpath_0 = join(path2outs, "{0}/img_{1}_{2}_{3}.exr".format(dataset, r, t, 0.0))
            imgpath_empty = join(path2outs, "{0}/img_{1}.exr".format(datasetempty, illu))

            img = cv2.imread(imgpath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            img_0 = cv2.imread(imgpath_0, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            img_empty = cv2.imread(imgpath_empty, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            # Get average luminance
            meanlum0 = img_0[mask_allobj3D == 1].mean()
            meanlum = img[mask_allobj3D == 1].mean()
            coefflum = meanlum / meanlum0

            img_amb_objfloor = img_0*(mask_allobj3D)*coefflum + img_empty*(mask_floor3D) + img * (1-mask_objfloor3D)

            cv2.imwrite(path2outs+"{0}/img_{1}_{2}_{3}_floorsphere.exr".format(dataset, r, t, illu), img_amb_objfloor.astype(np.float32))
            shutil.copy(path2outs+"{0}/img_{1}_{2}_{3}_ref.exr".format(dataset, r, t, illu), path2outs+"{0}/img_{1}_{2}_{3}_floorsphere_ref.exr".format(dataset, r, t, illu))

### nocue condition

#mask_rightsheet3D = cv2.imread(path2outs + f'masks_{dat}pattern/rightsheet_ref.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
mask_rightsheet3D = cv2.imread(path2outs + f'masks/rightsheetsmallcycles_ref.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) # The other size is too big and creates highlights at the borders
mask_rightsheet3D = cv2.cvtColor(mask_rightsheet3D , cv2.COLOR_BGR2RGB)

for i, illu in enumerate(intensities):
    for r in ref_reflectances:
        for t in test_reflectances:
            imgpath = join(path2outs, "{0}/img_{1}_{2}_{3}.exr".format(dataset, r, t, illu))
            imgpath_0 = join(path2outs, "{0}/img_{1}_{2}_{3}.exr".format(dataset, r, t, 0.0))
            img = cv2.imread(imgpath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            img_0 = cv2.imread(imgpath_0, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            img_amb_whole = img*(mask_rightsheet3D) + img_0 * (1-mask_rightsheet3D)

            cv2.imwrite(path2outs+"{0}/img_{1}_{2}_{3}_whole.exr".format(dataset, r, t, illu), img_amb_whole.astype(np.float32))
            shutil.copy(path2outs+"{0}/img_{1}_{2}_{3}_ref.exr".format(dataset, r, t, illu), path2outs+"{0}/img_{1}_{2}_{3}_whole_ref.exr".format(dataset, r, t, illu))


