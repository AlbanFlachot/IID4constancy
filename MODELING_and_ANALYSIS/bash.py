'''
Python file with the purpose of a shell script.

Examples are given for running 1 image through the model, running the training script, the test script and the analysis script
'''

import subprocess
import shlex
import os


### Command line to run 1 image through the model and save the prediction in output

imagep = 'example.exr'
arch = 'AlbanNetsupeeveepatterns'
command = f"python3 run_single_image.py --gamma 0 --arch {arch} --cuda --image {imagep} --checkpointdir checkpoints/sup_1_eeveepatterns --level 1 --nepoch0 57 --nb_channels=1"
args = shlex.split(command)
subprocess.call(args)


### Command to run several images through the model and save the predictions
'''
dataset = "TwoCubesBlender1282p9" # testset to run though
model = "eevee" # training set the model was trained on
instance = 1 # Training instance (1 --> 3)
command = f"python3 test_pipeline.py --gamma 0 --cuda --testset {dataset} --checkpointdir checkpoints/sup_{instance}_{model}patterns --imList imList_all.txt --testRoot test_outs --imagedir_out {dataset}_AlbanNetsup{model}patterns_{instance}  --level 1 --nepoch0 57 --nb_channels=1"%(i,i)
args = shlex.split(command)
subprocess.call(args)'''

### Command to train the model (not recommanded outside of a server, and would need to render a training set first)

dataset = "TwoCubesBlender128" # testset to run though
model = "eevee" # training set the model was trained on
instance = 1 # Training instance (1 --> 3)
command = f"python3 test_pipeline.py --gamma 0 --cuda --testset {dataset} --checkpointdir checkpoints/sup_{instance}_{model}patterns --imList imList_all.txt --testRoot test_outs --imagedir_out {dataset}_AlbanNetsup{model}patterns_{instance}  --level 1 --nepoch0 57 --nb_channels=1"
args = shlex.split(command)
subprocess.call(args)

