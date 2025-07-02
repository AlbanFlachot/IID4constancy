import os, shutil, tarfile
from os.path import join
import glob

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--imgdir', type=str,   default='dataset',   help='path to dataset')
args = parser.parse_args()

dist_arch_train = tarfile.open(join(args.imgdir, 'train.tar'), mode='w') ## create archive

list_tars = glob.glob(args.imgdir + '/train[0-9]*.tar')
list_scenes = glob.glob(args.imgdir + '/train[0-9]*_scenes.txt')
list_files = glob.glob(args.imgdir + '/train[0-9]*_files.txt')

### concatenate archives
for arch in list_tars:
    os.mkdir(join(args.imgdir, 'tmp'))
    print(f'Processing {arch}')
    try:
        tarf = tarfile.open(arch)
        tarf.extractall(join(args.imgdir, 'tmp')) # extract files in tmp dir
    except Exception:
        pass
    listfiles = os.listdir(join(args.imgdir, 'tmp')) # list of files.
    for file in listfiles:
        dist_arch_train.add(join(args.imgdir, 'tmp/'+file), arcname= file)
    tarf.close()
    shutil.rmtree(join(args.imgdir, 'tmp'))
dist_arch_train.close()

### concatenate txt files
with open(args.imgdir + f'/train_scenes.txt', 'w') as txt_scenes:
    for fname in list_scenes:
        with open(fname) as infile:
            for line in infile:
                if not 'nrm' in line:
                    txt_scenes.write(line)
                
with open(args.imgdir + f'/train_files.txt', 'w') as txt_files:
    for fname in list_files:
        with open(fname) as infile:
            for line in infile:
                txt_files.write(line)
