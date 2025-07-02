import os, sys, math, random, argparse, numpy as np
import bpy
import bmesh

import json
from math import radians
from mathutils import Matrix

print(os.getcwd())
import blender_util as bu
from lib_alban import *

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nobjects', type=int, default=14, help='number of objects in the scene')
parser.add_argument('--filename', type=str, default='testset/images/imgfix2', help='filename of rendered image')
parser.add_argument('--rngseed', type=int, default=-1, help='random number generator seed')
parser.add_argument('--pointangle', type=float, default=5.0, help='point light angle subtended (degrees)')
parser.add_argument('--objtypes', type=int, nargs='*', default=[0, 1, 2, 3, 4])
parser.add_argument('--textypes', type=int, nargs='*', default=[0, 1, 2, 3])

parser.add_argument('--reference', type=float, default=0.5)
parser.add_argument('--test', type=float, default=0.5)
parser.add_argument('--illu', type=float, default=0.5)
parser.add_argument('--renderer', type=str, default='CYCLES')
parser.add_argument('--samples', type=int, default=128)

try:
    argv = sys.argv[sys.argv.index('--') + 1:]
except:
    argv = ''
args = parser.parse_args(argv)

RGB_isoluminant = np.load('RGB_isoluminant.npy')

## convert 8 bit color value to 0-1

# generate a random greyscale rgb value

## add a material to an object, create a new diffuse material if none exists by the specified name


# seed rng
if args.rngseed >= 0:
    random.seed(args.rngseed)
else:
    random.seed()

# delete cube
bpy.data.objects.remove(bpy.data.objects['Cube'])

# set rendering parameters
render = bpy.context.scene.render
render.resolution_x = render.resolution_y = 256
render.resolution_percentage = 100

# initialize intrinsic rendering manager
irm = bu.Intrinsic(hdr=True)
irm.reflectance = True
irm.normals = True
irm.depth = False
irm.objectid = False
irm.max_bounces = 3

# initialize camera manager
camera = bu.Camera(distance=14.0, fov=50.0, bg=None)

# initialize lighting manager
light = bu.Light()


# create a plane
def plane(name='Plane', rgb=(0.4, 0.4, 0.4, 1.0)):
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0), scale=(1, 1, 1))
    p = bpy.context.active_object
    p.name = name
    m = bpy.data.materials.new(name + 'Material')
    m.diffuse_color = bu.expand_rgb(rgb)
    m.specular_intensity = 0.0
    p.data.materials.append(m)
    return p


# add background planes
# floor = plane( 'Floor', rgb=(0.4,0.5,0.6, 1) )
# Lwall = plane( 'LeftWall', rgb=(0.5,0.6,0.4) )

floor = plane('Floor', rgb=(0, 0, 0, 1))
Lwall = plane('LeftWall', rgb=(0, 0, 0, 1))
Lwall.rotation_euler[0] = math.radians(-90.0)
# Lwall.rotation_euler[2] = math.radians( random.uniform(-20.0,20.0 ) )
# Rwall = plane( 'RightWall', rgb=(0.6,0.4,0.5))
Rwall = plane('RightWall', rgb=(0, 0, 0, 1))
Rwall.rotation_euler[1] = math.radians(90.0)
# Rwall.rotation_euler[2] = math.radians( random.uniform(-20.0,20.0) )


# create objects
group_obj = []  # list of objects in scene


set_locationsnSize = np.load('testset/locs_top.npy')

for i in range(args.nobjects):
    # create object
    location = set_locationsnSize[:3, i]
    rotation = set_locationsnSize[9:, i]
    obj, _ = create_shape(location, set_locationsnSize[3, i], rotation=rotation, shape=set_locationsnSize[4, i])
    obj.name = 'object_%06d' % (i + 1)

    # create material
    name = 'object_%06d_material' % (i + 1)
    # rgb = set_locationsnSize[5:9, i]
    rgb = tuple([0,0,0,0])
    # texture = random.choice(args.textypes)
    bu.diffuse( obj=obj, rgb=rgb )

    obj.pass_index = i + 1
    obj.pass_index += 100 * 2
    group_obj.append(obj)


## Creating the scene objects

shape_size = 1.1
paper_size = 0.7


## Sphere
sphere = create_shape((6.3, 4.5, 4.2), 2.5,  shape = 0)
#sphere, texture, rgb = create_obj_albedo(sphere[0], args.textypes, 'sphere', weights = [0.1,0.4,0.4,0.1])
bu.diffuse( obj=sphere[0], rgb=tuple([0, 0, 0, 1]) )

# target cube
reference_cube = create_shape((5.4, 3.6, shape_size/2), shape_size, rotation = np.array([0,0,0]), shape = 1)
rangetex = range(0,10)
bu.diffuse( obj=reference_cube[0], rgb=tuple([0,0,0,1]) )

## reference cube
target_cube = create_shape((3.6, 5.4, shape_size/2), shape_size, rotation = np.array([0,0,0]), shape = 1)
rangetex = range(0,10)
bu.diffuse( obj=target_cube[0], rgb=tuple([0,0,0,1]) )

## target paper
reference_paper = create_obj('plane', 'referencepaper', (5.3, 3.5, shape_size + 0.0001), (0, 0, 0), size = paper_size)
#reference_paper = create_obj('plane', 'referencepaper', (5.4, 3.6, shape_size + 0.0001), (0, 0, 0), size = paper_size)

## reference paper
target_paper = create_obj('plane', 'targetpaper', (3.5+0.003, 5.5+0.001, shape_size + 0.0001), (0, 0, 0), size = paper_size-0.045)
#target_paper = create_obj('plane', 'targetpaper', (3.6, 5.4, shape_size + 0.0001), (0, 0, 0), size = paper_size)

## target paper
add_mat('targetpaper', int(1*255))

## reference paper
add_mat('referencepaper', int(0*255))
### render

# f = open('list_files.txt', 'w')


# render_defaultrotate_all_objects
camera = set_camera(camera)
light = set_light(light, args.illu)

# np.save('testset/locs_top.npy', set_locationsnSize2)

irm.filename = args.filename
irm.render(renderer=args.renderer, samples=args.samples)







