import os, sys, math, random, argparse, numpy as np
import bpy

print(os.getcwd())
import blender_util as bu
from lib_alban import *

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nobjects', type=int, default=14, help='number of objects in the scene')
parser.add_argument('--filename', type=str, default='testset/images/imgfix2', help='filename of rendered image')
parser.add_argument('--rngseed', type=int, default=-1, help='random number generator seed')
parser.add_argument('--pointangle', type=float, default=3.5, help='point light angle subtended (degrees)')
parser.add_argument('--objtypes', type=int, nargs='*', default=[0, 1, 2, 3, 4])
parser.add_argument('--textypes', type=int, nargs='*', default=[0, 1, 2, 3])

parser.add_argument('--reference', type=float, default=0.5)
parser.add_argument('--test', type=float, default=0.5)
parser.add_argument('--illus', type=float,  nargs='*', default=[0.0, 0.35, 0.75, 1.5, 3.0])
parser.add_argument('--renderer', type=str, default='CYCLES')
parser.add_argument('--samples', type=int, default=128)
parser.add_argument('--diff_lum', type=float, default=1)
parser.add_argument('--tex', type=int, default=0)

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
floor = plane('Floor', rgb=(0.4, 0.5, 0.7, 1))
Lwall = plane('LeftWall', rgb=(0.5, 0.7, 0.4))
Lwall.rotation_euler[0] = math.radians(-90.0)
# Lwall.rotation_euler[2] = math.radians( random.uniform(-20.0,20.0 ) )
Rwall = plane('RightWall', rgb=(0.7, 0.4, 0.5))
Rwall.rotation_euler[1] = math.radians(90.0)
# Rwall.rotation_euler[2] = math.radians( random.uniform(-20.0,20.0) )

# render_defaultrotate_all_objects
camera = set_camera(camera)
for illu in args.illus:
    light = set_light(light, illu)

    # np.save('testset/locs_top.npy', set_locationsnSize2)

    irm.filename = args.filename + f'/img_{illu}'
    irm.render(renderer=args.renderer, samples=args.samples)







