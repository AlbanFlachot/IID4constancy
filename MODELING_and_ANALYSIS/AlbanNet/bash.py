'''
Python file with the purpose of a shell script.
'''

import subprocess
import shlex
import os


'''
datasets = ['TwoCubesBlendereevee', 'TwoCubesBlender1024', 'TwoCubesBlender128']
for dataset in datasets:
    command = f"python3 Analysis_2Tables.py --path2outs {dataset}_IRNet_0 --nb_instances 2"
    args = shlex.split(command)
    subprocess.call(args)'''


dataset = 'TwoCubesBlender1282p9'


for i in range(1,4):
    '''
    command = f"python3 test_pipeline.py --gamma 0 --arch AlbanNet --cuda --testset {dataset} --checkpointdir checkpoints/sup_%i_eeveepatterns --imList imList_all.txt --testRoot test_outs --imagedir_out {dataset}_AlbanNetsupeeveepatterns_%i  --level 1 --nepoch0 57 --nb_channels=1"%(i,i)
    args = shlex.split(command)
    subprocess.call(args)


    command = f"python3 test_pipeline.py --gamma 0 --arch AlbanNet --cuda --testset {dataset} --checkpointdir checkpoints/sup_%i_128patterns --imList imList_all.txt --testRoot test_outs --imagedir_out {dataset}_AlbanNetsup128patterns_%i  --level 1 --nepoch0 57 --nb_channels=1" % (
    i, i)
    args = shlex.split(command)
    subprocess.call(args)'''

    '''command = f"python3 test_pipeline.py --gamma 0 --arch AlbanNet --cuda --testset {dataset} --checkpointdir checkpoints/sup_%i_1283p6patterns --imList imList_all.txt --testRoot test_outs --imagedir_out {dataset}_AlbanNetsup1283p6patterns_%i  --level 1 --nepoch0 57 --nb_channels=1" % (
        i, i)
    args = shlex.split(command)
    subprocess.call(args)'''

'''
command = f'python3 Analysis_2Tables.py --path2outs {dataset}_AlbanNetsupeeveepatterns_1 --nb_instances 4'
args = shlex.split(command)
subprocess.call(args)'''
'''
command = f'python3 Analysis_2Tables.py --path2outs {dataset}_AlbanNetsup1283p6patterns --nb_instances 4'
args = shlex.split(command)
subprocess.call(args)'''
'''
command = f'python3 Analysis_2Tables.py --path2outs {dataset}_AlbanNetsup128patterns --nb_instances 4'
args = shlex.split(command)
subprocess.call(args)'''
'''
command = f'python3 Analysis_1cube.py --path2outs {dataset}_greyworld_1 --nb_instances 4'
args = shlex.split(command)
subprocess.call(args)'''


'''
command = f'python3 Analysis_1cube_IRNet.py --path2outs {dataset}_IRNetpatterns_1 --nb_instances 4'
args = shlex.split(command)
subprocess.call(args)'''

imagep = '/home/alban/Documents/dataset_eeveetest/example.exr'
arch = 'AlbanNetsupeeveepatterns'
command = f"python3 run_single_image.py --gamma 0 --arch {arch} --cuda --image {imagep} --checkpointdir checkpoints/sup_1_eeveepatterns --level 1 --nepoch0 57 --nb_channels=1"
args = shlex.split(command)
subprocess.call(args)
'''
datasets = ['validation_eevee']
for dataset in datasets:
    arch = 'AlbanNetsupeeveepatterns'
    command = f"python3 -i validation_pipeline.py --gamma 0 --arch {arch} --cuda --testset {dataset} --checkpointdir checkpoints/sup_1_eeveepatterns --imList imList_normal_reduced.txt --testRoot test_outs --level 1 --nepoch0 57 --nb_channels=1"
    args = shlex.split(command)
    subprocess.call(args)'''