'''
Python file with the purpose of a shell script.

Examples are given for running 1 image through the model, running the training script, the test script and the analysis script
'''

import subprocess
import shlex
import os

from numpy.distutils.system_info import triplet

### Command line to run 1 image through the model and save the prediction in output

imagep = 'example.exr'
arch = 'AlbanNetsupeeveepatterns'
command = f"python3 run_single_image.py --gamma 0 --arch {arch} --cuda --image {imagep} --checkpointdir checkpoints/sup_1_eeveepatterns --level 1 --nepoch0 57 --nb_channels=1"
args = shlex.split(command)
subprocess.call(args)

### Command to train the model (not recommanded if not on a server, and would need to render a training set first)

'''
dataset = "path2dataset" # testset to run though
instance_run = 1 # Training instance (1 --> 3)
instance_name = f"{dataset}_{instance_run}"
command = f"python3 train_pipeline.py --trainset {dataset} --experiment {instance_name} --arch AlbanNet --nbepochs 60 --contraW 1000 --contraLoss triplet --nb_channels 1
--lrStep 10"
args = shlex.split(command)
subprocess.call(args)
'''

### Command to run several images through the model and save the predictions.
## Note that by default it uses cuda

'''
dataset = "path2testset" # testset to run though e.g. "TwoCubesBlender128"
instance_run = 1 # Training instance (1 --> 3)
instance_name = f"{dataset}_{instance_run}"
instance = 1 # Training instance (1 --> 3)
command = f"python3 test_pipeline.py --gamma 0 --cuda --testset {dataset} --checkpointdir checkpoints/{instance_name} --imList imList_all.txt --testRoot test_outs --imagedir_out {dataset}_{instance_name}  --level 1 --nepoch0 57 --nb_channels=1"
args = shlex.split(command)
subprocess.call(args)
'''


### Command to run the first level analysis on all the model's outputs

'''
dataset = "path2testset" # testset the model was evaluated on e.g. "TwoCubesBlender128"
nb_instances = 3 # 3 training runs in our case
instance_name = f"{dataset}_1"
command = f'python3 Analysis.py --path2outs {instance_name} --nb_instances {nb_instances + 1}}'
args = shlex.split(command)
subprocess.call(args)'''