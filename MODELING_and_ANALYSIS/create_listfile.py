import os
from os.path import join
import glob

#path = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/TwoTables'
#path = '/home/alban/Dropbox/mlab/users/alban/works/InvRend/projects/datasets/val_geo_eevee'
#path = '/home/alban/Dropbox/mlab/users/shared/jay_illusions/illusion_images'
#path = '/home/alban/Documents/mitsuba3/rendering_files_testset/outs/new_images'
path = '/home/alban/Documents/blender_testset/testset/images_4test'

listpaths = glob.glob(path + '/*_ref.exr')
#listpaths = glob.glob(path + '/*.exr')

listfiles = [addr.split('/')[-1] for addr in listpaths]

#print(listfiles)

# create full imList.txt
with open(join(path,'imList.txt'), 'w') as f:
    for file in listfiles:
        if 'mask' not in file:
            #f.write(file.replace('_albedo.exr', '.exr'))
            f.write(file.replace('_ref.exr', '.exr'))
            #f.write(file)
            f.write("\n")

# create normal imList.txt
with open(join(path,'imList_normal.txt'), 'w') as f:
    for file in listfiles:
        if ('mask' not in file) & ('cube' not in file) & ('sphere' not in file) & ('floor' not in file) & ('whole' not in file):
            #f.write(file.replace('_albedo.exr', '.exr'))
            f.write(file.replace('_ref.exr', '.exr'))
            #f.write(file)
            f.write("\n")
