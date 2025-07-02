import os, sys, math, random, argparse, numpy as np
import bpy
import bmesh
import json
from math import radians
from mathutils import Matrix
print(os. getcwd())
import blender_util as bu
from lib_alban import *


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nobjects', type=int,   default=14,   help='number of objects in the scene')
parser.add_argument('--filename', type=str,   default='testset/images/imgfix2', help='filename of rendered image')
parser.add_argument('--rngseed',  type=int,   default=-1,   help='random number generator seed')
parser.add_argument('--pointangle', type=float, default=3.5, help='point light angle subtended (degrees)')
parser.add_argument('--objtypes', type=int, nargs='*', default=[0, 1, 2, 3, 4])
parser.add_argument('--textypes', type=int, nargs='*', default=[0, 1, 2, 3])

parser.add_argument('--references', type=float,  nargs='*', default=[0.2, 0.4, 0.6])
parser.add_argument('--tests', type=float,  nargs='*', default=[0.00, 0.025,0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.715, 0.775, 0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975])
#parser.add_argument('--tests', type=float,  nargs='*', default=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
parser.add_argument('--illus', type=float,  nargs='*', default=[0.0, 0.35, 0.75, 1.5, 3.0])
parser.add_argument('--renderer', type=str,  default='CYCLES')
parser.add_argument('--samples', type=int,  default=128)
parser.add_argument('--diff_lum', type=float,  default=1)


try:
    argv = sys.argv[ sys.argv.index('--')+1: ]
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
bpy.data.objects.remove( bpy.data.objects['Cube'] )

# set rendering parameters
render = bpy.context.scene.render
render.resolution_x = render.resolution_y = 256
render.resolution_percentage = 100

# initialize intrinsic rendering manager
irm = bu.Intrinsic( hdr=True )
irm.reflectance = True
irm.normals = True
irm.depth = False
irm.objectid = False
irm.max_bounces = 3

# initialize camera manager
camera = bu.Camera( distance=14.0, fov=50.0, bg=None )

# initialize lighting manager
light = bu.Light()

# create a plane
def plane( name='Plane', rgb=(0.4,0.4,0.4,1.0) ):
    bpy.ops.mesh.primitive_plane_add( size=20, location=(0,0,0), scale=(1,1,1) )
    p = bpy.context.active_object
    p.name = name
    m = bpy.data.materials.new( name+'Material' )
    m.diffuse_color = bu.expand_rgb( rgb )
    m.specular_intensity = 0.0
    p.data.materials.append( m )
    return p

# add background planes
floor = plane( 'Floor', rgb=(0.4,0.5,0.8, 1) )
Lwall = plane( 'LeftWall', rgb=(0.5,0.8,0.4) )
Lwall.rotation_euler[0] = math.radians(-90.0)
#Lwall.rotation_euler[2] = math.radians( random.uniform(-20.0,20.0 ) )
Rwall = plane( 'RightWall', rgb=(0.8,0.4,0.5))
Rwall.rotation_euler[1] = math.radians(90.0)
#Rwall.rotation_euler[2] = math.radians( random.uniform(-20.0,20.0) )


# create objects
group_obj = [] # list of objects in scene
locNsizes = [] # list of locations and sizes of objects


set_locationsnSize = np.load('testset/locs_top.npy')

textures = [0,1,0,3,2,0,0,0,0,2,3,0,0,3]

for i in range(args.nobjects):

    # create object
    location = set_locationsnSize[:3, i]
    rotation = set_locationsnSize[9:, i]
    obj, _ = create_shape(location, set_locationsnSize[3,i], rotation = rotation, shape = set_locationsnSize[4,i])
    obj.name = 'object_%06d' % ( i + 1 )

    # create material
    name= 'object_%06d_material' % ( i + 1 )
    rgb = set_locationsnSize[5:9, i]
    #texture = random.choice(args.textypes)
    #obj, rgb = create_obj_albedo(obj, rgb, name)
    obj, texture, rgb = create_obj_albedo(obj, args.textypes, name, texture = textures[i], rgb = rgb, pattern= f'testset/patterns/color_pattern{i-2}.jpg')

    obj.pass_index = i+1
    obj.pass_index += 100 * 2
    group_obj.append(obj)


'''
set_locationsnSize2 = np.zeros((12,args.nobjects))
set_locationsnSize2[:5] = set_locationsnSize
set_locationsnSize2[5:9] = np.array(rgbs).T
set_locationsnSize2[9:] = np.array(rotations).T'''

## Creating the scene objects

shape_size = 1.1
paper_size = 0.7

## Sphere
sphere = create_shape((6.3, 4.5, 4.2), 2.5,  shape = 0)
#sphere, texture, rgb = create_obj_albedo(sphere[0], args.textypes, 'sphere', weights = [0.1,0.4,0.4,0.1])
bu.diffuse( obj=sphere[0], rgb=tuple([0.65, 0.65, 0.65, 1]) )

# target cube
reference_cube = create_shape((5.4, 3.6, shape_size/2), shape_size, rotation = np.array([0,0,0]), shape = 1)
rangetex = range(0,10)
#bu.patterns(obj = target_cube[0], tex2apply = f'testset/patterns/pattern{np.random.choice(rangetex, 1)[0]}.jpg')
bu.patterns(obj = reference_cube[0], tex2apply = 'testset/patterns/pattern5.jpg')


## reference cube

target_cube = create_shape((3.6, 5.4, shape_size/2), shape_size, rotation = np.array([0,0,0]), shape = 1)
rangetex = range(0,10)
#bu.patterns(obj = reference_cube[0], tex2apply = f'testset/patterns/pattern{np.random.choice(rangetex, 1)[0]}.jpg', difflum=args.diff_lum)
bu.patterns(obj = target_cube[0], tex2apply = 'testset/patterns/pattern3.jpg', difflum=args.diff_lum)

## target paper
reference_paper = create_obj('plane', 'referencepaper', (5.3, 3.5, shape_size + 0.0001), (0, 0, 0), size = paper_size)
#reference_paper = create_obj('plane', 'referencepaper', (5.4, 3.6, shape_size + 0.0001), (0, 0, 0), size = paper_size)

## reference paper
target_paper = create_obj('plane', 'targetpaper', (3.5, 5.5, shape_size + 0.0001), (0, 0, 0), size = paper_size)
#target_paper = create_obj('plane', 'targetpaper', (3.6, 5.4, shape_size + 0.0001), (0, 0, 0), size = paper_size)

## reference cube

## patterns on floor
camera = set_camera(camera)

for ref in args.references:
    ## target paper
    add_mat('referencepaper', int(ref * 255))
    for test in args.tests:
        ## reference paper
        add_mat('targetpaper', int(test * 255))
        for i, illu in enumerate(args.illus):
            light = set_light(light, illu)

            ### render
            filename = args.filename + f'/img_{ref}_{test}_{illu}'
            irm.filename = filename
            irm.render(renderer = args.renderer, samples = args.samples)

# cube
list_diff_lum = [1.0175456, 1.445243, 1.93404, 2.850535, 4.683524]
for ref in args.references:
    ## target paper
    add_mat('referencepaper', int(ref * 255))
    for test in args.tests:
        ## reference paper
        add_mat('targetpaper', int(test * 255))
        for i, illu in enumerate(args.illus):
            light = set_light(light, illu)
            bu.patterns(obj=target_cube[0], tex2apply=f'testset/patterns/pattern3_{i}.jpg')
            ### render
            filename = args.filename + f'/img_{ref}_{test}_{illu}_cube'
            irm.filename = filename
            irm.render(renderer = args.renderer, samples = args.samples)







