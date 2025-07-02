import subprocess
import shlex
import util, slurm
import os


nb_gpus = 20 # Number of GPUs used to render the dataset
nb_scenes = 15000 # Each scene leads to 9 images --> roughly 121000 training images, 7000 validation images and 7000 test images
rendertime = 75 # Number of seconds to render 9 images. Above and the rendering is skipped (happens less than 1% times)
renderer = 'EEVEE'
counts = 128 # Sample count in case the renderer is CYCLES
imagedir = 'dataset_eevee' # Name of the dataset.
scriptfile = 'render.py'

if not os.path.exists(imagedir):
	os.mkdir(imagedir)

#### Initialize the txt files

with open(imagedir + '/val_scenes.txt', 'w') as file:
    file.write('')
with open(imagedir + '/test_scenes.txt', 'w') as file:
    file.write('')


with open(imagedir + '/val_files.txt', 'w') as file:
    file.write('')
with open(imagedir + '/test_files.txt', 'w') as file:
    file.write('')

#### Initialize the archives
import tarfile

archiv = tarfile.open(imagedir+ '/test.tar', 'w')
archiv.close() 
archiv = tarfile.open(imagedir+ '/val.tar', 'w')
archiv.close() 


for gpu in range(nb_gpus):
    ### Deciding which archive and txt files to fill up
    if gpu == (nb_gpus - 2):
        archivtype = 'val'
    elif gpu == (nb_gpus - 1):
        archivtype = 'test'
    else:
        archiv = tarfile.open(imagedir+ f'/train{gpu}.tar', 'w')
        archiv.close()
        with open(imagedir + f'/train{gpu}_files.txt', 'w') as file:
            file.write('')
        with open(imagedir + f'/train{gpu}_scenes.txt', 'w') as file:
            file.write('')
        archivtype = f'train{gpu}'
    
    # Set arguments
    nb2render = nb_scenes//nb_gpus
    scene_start = nb2render*gpu
    scene_end = scene_start + nb2render
    totaltime = nb2render*rendertime 
    
    # Start process
    command = f"sbatch --time={slurm.sec2time(totaltime)} job_render.sh {scene_start} {scene_end-1} {imagedir} {scriptfile} {renderer} {counts} {archivtype}"
    args = shlex.split(command)
    subprocess.call(args)
