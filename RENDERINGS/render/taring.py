import os
import tarfile
from os.path import join
import glob
import argparse

versions = 9 # 3lights*3albedos
intrinsics = 3 # rendering; albedo; normals

parser = argparse.ArgumentParser()

parser.add_argument('--set', type=str,   default='train',   help='should add to train, val or test datasets')
parser.add_argument('--scene', type=str,   default='image00000',   help='which scene are we looking to add')
parser.add_argument('--imgdir', type=str,   default='dataset',   help='path to dataset')

args = parser.parse_args()

archiv = tarfile.open(join(args.imgdir, args.set + '.tar'), mode='a') ## append to existing archive

listimgs = glob.glob(join(args.imgdir,args. scene+'*'))


#### Initialize appending to the txt files
txt_scenes = open(args.imgdir + f'/{args.set}_scenes.txt', 'a') # append
txt_files = open(args.imgdir + f'/{args.set}_files.txt', 'a')


for fil in listimgs:
    if len(listimgs) == versions*intrinsics: # only if the rendering was complete do we add it to the archive
        archiv.add(fil, arcname= fil.split('/')[-1])
        if (not "ref" in fil.split('/')[-1]) and (not "nrm" in fil.split('/')[-1]): # ignore gt images for the txt list
            txt_files.write(fil.split('/')[-1] + "\n") # Write the scene in list
    os.remove(fil) # delete files that just got archived (or not archived if incomplete)

txt_files.close()
archiv.close()

if len(listimgs) == versions*intrinsics:
    txt_scenes.write(args.scene + '\n') # Write the files in list 

txt_scenes.close()
