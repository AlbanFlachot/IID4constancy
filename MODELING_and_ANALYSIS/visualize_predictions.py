import os
import cv2

from lib_alban.utils_analyse import *


dataset = 'TwoCubesBlendereevee2p9'
model = 'AlbanNetsupeeveepatterns'
savename = f'{dataset}_{model}_3'
path2preds = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/pipeline_intrinsic_3datasets/test_outs/results_%s'%(savename)
#path2preds = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/pipeline_intrinsic_3datasets/test_outs/img-thouless-eevee-out/image_out'
path2datasets = '/home/alban/Documents/blender_testset/testset/images_cubespattern_eevee2p9'
#path2dataset = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/blender_eevee_patterns'


list_labels_ref, list_labels_illu, list_labels_match, list_labels_leftright = labels(dataset)
list_conditions = ['', '_cube', '_sphere', '_floor',  '_floorsphere', '_whole']

ref0 = 0.6
illu0 = 3.0
condition0 = ''
test0 = 0.6

nb_rows = 5
nb_cols = 5
print(list_labels_match)


fig, subs = plt.subplots(nb_rows, nb_cols, figsize = (2*nb_cols,2*nb_rows))
for c, Rmatch in enumerate(list_labels_match[2:7]):
    path = "img_{0}_{1}_{2}{3}.exr".format( ref0, Rmatch, illu0, condition0)
    print(join(path2preds,path))
    im = load_and_process_exr(join(path2datasets,path))
    albedo_pred = load_and_process_exr(join(path2preds,path)[:-4] + '_albedo0.exr')
    albedo_ref = load_and_process_exr(join(path2datasets, path)[:-4] + '_ref.exr')
    illu_pred = load_and_process_exr(join(path2preds, path)[:-4] + '_illu.exr')
    illu_ref = load_and_process_exr(join(path2preds, path)[:-4] + '_illuref.exr')
    subs[0, c].imshow((im[:,:,0]**(1/2.2)), cmap='gray', vmin=0, vmax=1)
    subs[0,c].axis('off')
    subs[2, c].imshow(albedo_pred[:,:,0], cmap='gray', vmin=0, vmax=1)
    subs[2, c].axis('off')
    subs[1, c].imshow(albedo_ref[:,:,0], cmap='gray', vmin=0, vmax=1)
    subs[1, c].axis('off')
    subs[4, c].imshow(illu_pred.mean(-1), cmap='gray', vmin=0, vmax=1)
    subs[4, c].axis('off')
    subs[3, c].imshow(illu_ref.mean(-1), cmap='gray', vmin=0, vmax=1)
    subs[3, c].axis('off')
fig.tight_layout()
fig.savefig(f'figures/{model}/' + savename + f'_examples_tests{condition0}.png')
plt.show()
plt.close()


nb_cols = len(list_labels_illu)
fig, subs = plt.subplots(nb_rows, nb_cols, figsize = (2*nb_cols,2*nb_rows))
for c, illu in enumerate(list_labels_illu):
    path = "img_{0}_{1}_{2}{3}.exr".format( ref0, test0, illu, condition0)
    im = load_and_process_exr(join(path2datasets,path))
    albedo_pred = load_and_process_exr(join(path2preds,path)[:-4] + '_albedo0.exr')
    albedo_ref = load_and_process_exr(join(path2preds, path)[:-4] + '_ref.exr')
    illu_pred = load_and_process_exr(join(path2preds, path)[:-4] + '_illu.exr')
    illu_ref = load_and_process_exr(join(path2preds, path)[:-4] + '_illuref.exr')
    subs[0, c].imshow((im[:,:,0]**(1/2.2)), cmap='gray', vmin=0, vmax=1)
    subs[0,c].axis('off')
    subs[2, c].imshow(albedo_pred[:,:,2], cmap='gray', vmin=0, vmax=1)
    subs[2, c].axis('off')
    subs[1, c].imshow(albedo_ref[:,:,2], cmap='gray', vmin=0, vmax=1)
    subs[1, c].axis('off')
    subs[4, c].imshow(illu_pred.mean(-1), cmap='gray', vmin=0, vmax=1)
    subs[4, c].axis('off')
    subs[3, c].imshow(illu_ref.mean(-1), cmap='gray', vmin=0, vmax=1)
    subs[3, c].axis('off')
subs[0, 0].set_label('Input')
subs[1, 0].set_label('Ground Truth')
subs[2, 0].set_label('Prediction')
fig.tight_layout()
fig.savefig(f'figures/{model}/' + savename + f'_examples_illus{condition0}.png')
plt.show()

ref = 0.4
illu = 1.5
condition = ''
test = 0.4


nb_cols = len(list_conditions)
fig, subs = plt.subplots(nb_rows, nb_cols, figsize = (2*nb_cols,2*nb_rows))
for c, condition in enumerate(list_conditions):
    path = "img_{0}_{1}_{2}{3}.exr".format( ref0, test0, illu0, condition)
    print(path)
    im = load_and_process_exr(join(path2datasets,path))
    albedo_pred = load_and_process_exr(join(path2preds,path)[:-4] + '_albedo0.exr')
    albedo_ref = load_and_process_exr(join(path2preds, path)[:-4] + '_ref.exr')
    illu_pred = load_and_process_exr(join(path2preds, path)[:-4] + '_illu.exr')
    illu_ref = load_and_process_exr(join(path2preds, path)[:-4] + '_illuref.exr')
    subs[0, c].imshow((im[:,:,2]**(1/2.2)), cmap='gray', vmin=0, vmax=1)
    subs[0,c].axis('off')
    subs[2, c].imshow(albedo_pred[:,:,2], cmap='gray', vmin=0, vmax=1)
    subs[2, c].axis('off')
    subs[1, c].imshow(albedo_ref[:,:,2], cmap='gray', vmin=0, vmax=1)
    subs[1, c].axis('off')
    subs[4, c].imshow(illu_pred.mean(-1), cmap='gray', vmin=0, vmax=1)
    subs[4, c].axis('off')
    subs[3, c].imshow(illu_ref.mean(-1), cmap='gray', vmin=0, vmax=1)
    subs[3, c].axis('off')
subs[0, 0].set_label('Input')
subs[1, 0].set_label('Ground Truth')
subs[2, 0].set_label('Prediction')
fig.tight_layout()
fig.savefig(f'figures/{model}/' + savename + '_examplesconditions.png')
plt.show()

# Display cue x lighting conditions
nb_cols = len(list_labels_illu)
nb_rows = len(list_conditions)
fig, subs = plt.subplots(nb_rows, nb_cols, figsize = (2*nb_cols,2*nb_rows))
for i, illu in enumerate(list_labels_illu):
    for c, condition in enumerate(list_conditions):
        path = "img_{0}_{1}_{2}{3}.exr".format( ref0, test0, illu, condition)
        print(path)
        im = load_and_process_exr(join(path2datasets,path))
        albedo_pred = load_and_process_exr(join(path2preds,path)[:-4] + '_albedo0.exr')
        albedo_ref = load_and_process_exr(join(path2preds, path)[:-4] + '_ref.exr')
        illu_pred = load_and_process_exr(join(path2preds, path)[:-4] + '_illu.exr')
        illu_ref = load_and_process_exr(join(path2preds, path)[:-4] + '_illuref.exr')
        subs[c, i].imshow((im[:,:,2]**(1/2.2)), cmap='gray', vmin=0, vmax=1)
        subs[c, i].axis('off')
subs[0, 0].set_label('Input')
subs[1, 0].set_label('Ground Truth')
subs[2, 0].set_label('Prediction')
fig.tight_layout()
fig.savefig(f'figures/examplesconditions.png')
plt.show()

'''
#### Display validation images
valpath = '/home/alban/Documents/validation_1283p6'
with open(valpath  + '/displist.txt', 'r') as f:
    valout = f.readlines()

nb_cols = 5
nb_rows = 3
fig, subs = plt.subplots(nb_rows, nb_cols, figsize = (2*nb_cols,2*nb_rows))
for c in range(nb_cols):
    path = f'{valpath}/outputs/{valout[c]}'
    print(path)
    im = load_and_process_exr(path[:-1])
    albedo_pred = load_and_process_exr(path[:-5] + '_albedo0.exr')
    albedo_ref = load_and_process_exr(path[:-5] + '_ref.exr')
    subs[0, c].imshow((im[:,:,2]**(1/2.2)), cmap='gray', vmin=0, vmax=1)
    subs[0,c].axis('off')
    subs[2, c].imshow(albedo_pred[:,:,2], cmap='gray', vmin=0, vmax=1)
    subs[2, c].axis('off')
    subs[1, c].imshow(albedo_ref[:,:,2], cmap='gray', vmin=0, vmax=1)
    subs[1, c].axis('off')
#subs[0, 0].set_label('Input')
#subs[1, 0].set_label('Ground Truth')
#subs[2, 0].set_label('Prediction')
fig.tight_layout()
fig.savefig('figures/paper/validation_examples.png')
plt.show()'''