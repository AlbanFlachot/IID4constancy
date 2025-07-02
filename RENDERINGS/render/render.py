import os, sys, math, random, argparse, numpy as np
import bpy
import bmesh
import blender_util as bu
import json
from math import radians
from mathutils import Matrix


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nobjects', type=int,   default=14,   help='number of objects in the scene')
parser.add_argument('--filename', type=str,   default='testset/test', help='filename of rendered image')
parser.add_argument('--rngseed',  type=int,   default=-1,   help='random number generator seed')
parser.add_argument('--pointangle', type=float, default=5.0, help='point light angle subtended (degrees)')
parser.add_argument('--objtypes', type=int, nargs='*', default=[0, 1, 2, 3, 4])
parser.add_argument('--textypes', type=int, nargs='*', default=[0, 1, 2, 3])

parser.add_argument('--reference', type=float,  default=0.5)
parser.add_argument('--test', type=float,  default=0.5)
parser.add_argument('--illu', type=float,  default=2)
parser.add_argument('--renderer', type=str,  default='CYCLES')
parser.add_argument('--samples', type=int,  default=1024)
parser.add_argument('--diff_lum', type=float,  default=1)

try:
    argv = sys.argv[ sys.argv.index('--')+1: ]
except:
    argv = ''
args = parser.parse_args(argv)

# generate a random xyz location
def rxyz( scale = 1 ):
    return( tuple( [ random.uniform( 0, scale ) for i in range(3) ] ) )

RGB_isoluminant = np.load('RGB_isoluminant.npy')

# generate a random greyscale rgb value
def rrgb( min=0.05, max=0.90, grey=False, light=False ):
    if grey:
        return 3*(random.uniform(min,max),) + (1.0,)
    else:
        if light:
            return random.choice(RGB_isoluminant)
        else:
            return tuple( random.uniform(min,max) for i in range(3) ) + (1.0,)
        
def rotate_object(obj, angle_degrees, axis='Z'):
    ''' rotates an object '''

    # local rotation about axis
    obj.rotation_euler = (obj.rotation_euler.to_matrix() @ Matrix.Rotation(radians(angle_degrees), 3, axis)).to_euler()


def set_camera(camera, reset = False, default = False, params = {}):
	if reset: # revert to default parameters computed previously
		camera.azimuth = params['azimuth']
		camera.elevation = params['elevation']
		camera.camera.location[1] = params['location'][1]
		camera.camera.location[2] = params['location'][2]
		
	else: # set random parameters and save them
		if default:
			camera.azimuth = random.uniform( 30.0, 60.0 )
			camera.elevation = random.uniform( 10.0, 40.0 )
			camera.camera.location[1] = random.uniform( -0.5, 0.5 )
			camera.camera.location[2] = random.uniform( -0.5, 0.5 )
			#camera.look_at()
			params={}
			params['azimuth'] = camera.azimuth
			params['elevation'] = camera.elevation
			params['location'] = camera.location[:]
		else:
			camera.azimuth = random.uniform( 20.0, 70.0 )
			camera.elevation = random.uniform( 10.0, 60.0 )
			camera.camera.location[1] = random.uniform( -0.5, 0.5 )
			camera.camera.location[2] = random.uniform( -0.5, 0.5 )
	return camera, params

def set_light(light, reset = False, params={}):
	if reset: # revert to default parameters computed previously
		light.ambient = params['ambient']
		light.angle = params['angle']
		light.point = params['point']
		light.azimuth = params['azimuth']
		light.elevation = params['elevation']
	else: # set random parameters and save them
		light.ambient = rrgb(light=True), random.uniform(0.15, 0.30)
		light.angle = random.uniform(2, 5)
		light.point = rrgb(light=True), random.uniform(0, 6)
		light.azimuth = random.uniform(20.0, 70.0)
		light.elevation = random.uniform(10.0, 85.0)
		params={}
		params['ambient'] = light.ambient
		params['angle'] = light.angle
		params['point'] = light.point
		params['azimuth'] = light.azimuth
		params['elevation'] = light.elevation
	return light, params

def rotate_all_objects(group_obj):
	for obj in group_obj:
	    if 100 < obj.pass_index < 300:
	    	continue
	    else:
		    xangle = random.uniform( 0, math.pi*2 )
		    yangle = random.uniform( 0, math.pi*2 )
		    zangle = random.uniform( 0, math.pi*2 )
		    obj.rotation_euler = (obj.rotation_euler.to_matrix() @ Matrix.Rotation(zangle, 3, 'Z')).to_euler()
		    obj.rotation_euler = (obj.rotation_euler.to_matrix() @ Matrix.Rotation(yangle, 3, 'Y')).to_euler()
		    obj.rotation_euler = (obj.rotation_euler.to_matrix() @ Matrix.Rotation(xangle, 3, 'X')).to_euler()
	return group_obj

def create_obj_albedo(obj, textypes, name, weights = [0.5, 0.1, 0.1, 0.3]):
	# create material
	texture = random.choices(args.textypes, weights = weights)[0]
	rgb = rrgb()
	if texture==0:
		bu.diffuse( obj=obj, rgb=rgb, name=name )
	elif texture==1:
		bu.voronoi( obj=obj, scale=random.uniform(1.0,10.0), name=name )
	elif texture==2:
		bu.noise( obj=obj, name=name )
	elif texture==3:
		bu.patterns( obj=obj, name=name )
	#print(name)
	return obj, texture, rgb

def create_diffuse_albedo(obj, rgb, name):
	# create material
	bu.diffuse( obj=obj, rgb=rgb, name=name )
	return obj

def create_shape(location, size):
    rotation = tuple( random.uniform(0,2*math.pi) for i in range(3) )
    shape = random.choice(args.objtypes)
    if shape==0:
        bpy.ops.mesh.primitive_ico_sphere_add( location=location, radius=size/2, rotation=rotation, subdivisions=4 )
    elif shape==1:
        bpy.ops.mesh.primitive_cube_add( location=location, size=size, rotation=rotation )
    elif shape==2:
        bpy.ops.mesh.primitive_ico_sphere_add( location=location, radius=size/2, rotation=rotation, subdivisions=1 )
    elif shape==3:
        bpy.ops.mesh.primitive_cylinder_add( location=location, radius=size/2, depth=size, rotation=rotation, vertices=128 )
    elif shape==4:
        bpy.ops.mesh.primitive_cone_add( location=location, radius1=size/2, radius2=0, depth=size, rotation=rotation, vertices=128 )
    return bpy.context.active_object
    
def create_mesh(size, obj):
	#import pdb; pdb.set_trace()
	rotation = tuple( random.uniform(0,2*math.pi) for i in range(3) )
	shape = random.choice(args.objtypes)
	# Create an empty mesh and the object.
	
	# Construct the bmesh cube and assign it to the blender mesh.
	bm = bmesh.new()
	if shape==0:
		bmesh.ops.create_icosphere(bm, subdivisions=4, diameter=2*size/3)
	elif shape==1:
		bmesh.ops.create_cube(bm, size = size)
	elif shape==2:
		bmesh.ops.create_icosphere(bm, subdivisions=1, diameter=2*size/3)
	elif shape==3:
		bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments = 128, diameter1=2*size/3, diameter2=2*size/3, depth=size)
	elif shape==4:
		bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments = 128, diameter1=2*size/3, diameter2=0, depth=size)
	bm.to_mesh(obj.data)
	bm.free()
	
	return obj
 
def modify_shapes(group_obj, locNsizes, rgbs):
	group_obj_new = []
	for i, obj in enumerate(group_obj):
		rotation = tuple( random.uniform(0,360) for i in range(3) )
		shape = random.choice(args.objtypes)
		#import pdb; pdb.set_trace()
		
		if 100 < obj.pass_index < 300:
			continue
		else:
			obj = create_mesh(locNsizes[i][1], obj)
			rotate_object(obj, rotation[0], axis='X')
			rotate_object(obj, rotation[1], axis='Y')
			rotate_object(obj, rotation[2], axis='Z')
			
	return group_obj_new

def reset_obj():
	for i, obj in enumerate(group_obj):
		obj =  group_obj_default[i].copy()
	return group_obj

def save_json(name):
	''' Function that saves metadata of each image'''
	# data to be written (dict)
	dict2save={}
	params_cam = {}
	params_cam['azimuth'] = camera.azimuth
	params_cam['elevation'] = camera.elevation
	params_cam['location'] = camera.location[:]
	dict2save['camera'] = params_cam
	
	params_l={}
	params_l['ambient'] = light.ambient
	params_l['angle'] = light.angle
	params_l['point'] = light.point
	params_l['azimuth'] = light.azimuth
	params_l['elevation'] = light.elevation
	dict2save['light'] = params_l
	
	# Serializing json
	json_object = json.dumps(dict2save, indent=4)
	
	# writing file
	with open(name + ".json", "w") as outfile:
   		outfile.write(json_object)



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
floor = plane( 'Floor', rgb=rrgb() )
Lwall = plane( 'LeftWall', rgb=rrgb() )
Lwall.rotation_euler[0] = math.radians(-90.0)
Lwall.rotation_euler[2] = math.radians( random.uniform(-20.0,20.0 ) )
Rwall = plane( 'RightWall', rgb=rrgb() )
Rwall.rotation_euler[1] = math.radians(90.0)
Rwall.rotation_euler[2] = math.radians( random.uniform(-20.0,20.0) )

if np.random.random(1)[0] < 0.25:
	bu.patterns(obj=floor, name='FloorMaterial')
if np.random.random(1)[0] < 0.25:
	bu.patterns(obj=Lwall, name='LeftWallMaterial')
if np.random.random(1)[0] < 0.25:
	bu.patterns(obj=Rwall, name='RightWallMaterial')

# create objects
group_obj = [] # list of objects in scene
locNsizes = [] # list of locations and sizes of objects
rgbs = [] # list of albedo of ojects (used for diffuse)
for i in range(args.nobjects):

    # create object
    location = rxyz(6)
    size = random.uniform(0.4,1.9)
    obj = create_shape(location, size)
    obj.name = 'object_%06d' % ( i + 1 )

    # create material
    name= 'object_%06d_material' % ( i + 1 )
    #texture = random.choice(args.textypes)
    obj, texture, rgb = create_obj_albedo(obj, args.textypes, name)

    obj.pass_index = i+1
    obj.pass_index += 100 * texture
    group_obj.append(obj)
    locNsizes.append([location, size])
    rgbs.append(rgb)


### render

#f = open('list_files.txt', 'w')
    

# render_defaultrotate_all_objects
camera, default_camera = set_camera(camera, default = True)
light, default_light = set_light(light)


irm.filename = args.filename +  '_camera%i_light%i_rot%i_alb%i'%(0, 0, 0, 0)
#irm.render()
#save_json(irm.filename)


group_obj_default = group_obj.copy()

#import pdb; pdb.set_trace()
'''
for trans in range(3,-1, -1):
	#print('What is happening here???')
	#print(camera.location)
	group_obj = modify_shapes(group_obj_default, locNsizes, rgbs)
	irm.filename = args.filename + '_camera%i_light%i_rot%i_alb%i'%(0, 0, trans, 0)
	irm.render()
	save_json(irm.filename)
	

#group_obj = reset_obj()

for cam in range(1,4):
	camera, _ = set_camera(camera)
	irm.filename = args.filename + '_camera%i_light%i_rot%i_alb%i'%(cam,0, 0, 0)
	irm.render()
	save_json(irm.filename)'''


#camera, _ = set_camera(camera, reset = True, params = default_camera)



for alb in range(1,4):
	for obj in group_obj_default:
		#import pdb; pdb.set_trace()
		idx = obj.pass_index%100
		obj, _, __ = create_obj_albedo(obj, args.textypes, 'object_%06d_material' % ( idx ), weights = [0.5, 0.1, 0.1, 0.3])
	if np.random.random(1)[0]>0.25:
		bu.diffuse( obj=floor, rgb=rrgb(), name='FloorMaterial' )
	else:
		bu.patterns( obj=floor, name='FloorMaterial')
	if np.random.random(1)[0]>0.25:
		bu.diffuse( obj=Lwall, rgb=rrgb(), name='LeftWallMaterial' )
	else:
		bu.patterns( obj=Lwall, name='LeftWallMaterial')
	if np.random.random(1)[0]>0.25:
		bu.diffuse( obj=Rwall, rgb=rrgb(), name='RightWallMaterial' )
	else:
		bu.patterns( obj=Rwall, name='RightWallMaterial')

	for li in range(1, 4):
		if (alb) == 1: # saving all 3 light params for next albodos
			if li == 1:
				light, params1 = set_light(light)
			elif li == 2:
				light, params2 = set_light(light)
			elif li == 3:
				light, params3 = set_light(light)
		else: # applying previously set light params to	 next albodos
			if li == 1:
				light, _ = set_light(light, reset = True, params=params1)
			elif li == 2:
				light, _ = set_light(light, reset = True, params=params2)
			elif li == 3:
				light, _ = set_light(light, reset = True, params=params3)
		irm.filename = args.filename + '_camera%i_light%i_rot%i_alb%i' % (0, li, 0, alb)
		irm.render(renderer=args.renderer, samples=args.samples)
		#save_json(irm.filename)





light, _ = set_light(light, reset = True, params = default_light)
#group_obj = reset_obj()



