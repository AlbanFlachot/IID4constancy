import os, sys, math, random, argparse, numpy as np
import bpy
import bmesh
import blender_util as bu
import json
from math import radians
from mathutils import Matrix

obj_types = [0, 1, 2, 3, 4]
textypes = [0, 1, 2, 3]
pointangle = 3.5

def angle_rad(x):
    return np.dot(x, (np.pi/180))

## check if an object exist with the name specified
def check_obj(name):
    return bpy.data.objects.get(name)

## convert 8 bit color value to 0-1
def color_255(x):
    i = np.linspace(0, 1, 256)
    return (i[x], i[x], i[x], 1)

## remove the specified object 
def rem_obj(name):
    ob = bpy.data.objects.get(name)
    if ob is not None:
        ob.select_set(True)
        bpy.ops.object.delete()

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


def set_camera(camera):
		
	camera.azimuth = 45# random.uniform( 30.0, 60.0 )
	camera.elevation = 25# random.uniform( 10.0, 40.0 )
	camera.camera.location[1] = 0#random.uniform( -0.5, 0.5 )
	camera.camera.location[2] = 0#random.uniform( -0.5, 0.5 )
	return camera

def set_light(light, illu):
	light.ambient = ( 1.0,1.0,1.0), 0.25#random.uniform(0.15, 0.30)
	light.angle = 3.5
	light.point = ( 1.0,1.0,1.0 ), illu#random.uniform(0.25, 6)
	light.azimuth = 45#random.uniform(20.0, 70.0)
	light.elevation = 70#random.uniform(10.0, 85.0)
	return light

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

def check_obj(name):
    return bpy.data.objects.get(name)

def create_obj(type, name, location, rotation = [0, 0, 0], size = 0.5, scale = (1, 1, 1)):
    
    ob = check_obj(name)
    if type == 'plane':
        if ob is None:
            ob = bpy.ops.mesh.primitive_plane_add(size = size, scale = scale)
        else:
            return ob
        
    elif type == 'cube':
        if ob is None:
            ob = bpy.ops.mesh.primitive_cube_add(size = size, scale = scale)
        else:
            return ob

    elif type == 'circle':
        if ob is None:
            ob = bpy.ops.mesh.primitive_circle_add(radius = size, scale = scale)
        else:
            return ob

    elif type == 'cylinder':
        if ob is None:
            ob = bpy.ops.mesh.primitive_cylinder_add(scale = scale)
        else:
            return ob

    elif type == 'torus':
        if ob is None:
            ob = bpy.ops.mesh.primitive_torus_add(major_radius = size[0], minor_radius = size[1])
        else:
            return ob

    elif type == 'cone':
        if ob is None:
            ob = bpy.ops.mesh.primitive_cone_add(scale = scale)
        else:
            return ob
    
    elif type == 'sphere':
        if ob is None:
            ob = bpy.ops.mesh.primitive_uv_sphere_add(radius = size, scale = scale)
        else:
            return ob

    elif type == 'camera':
        if ob is None:
            ob = bpy.ops.object.camera_add(align = 'VIEW', scale = scale)
        else:
            return ob

    elif type == 'light':
        if ob is None:
            ob = bpy.ops.object.light_add(type = 'SPOT', align = 'WORLD', scale = scale)

    ob = bpy.context.object
    ob.name = name
    ob.location = location
    ob.rotation_euler = angle_rad(rotation)
    ob.display_type = 'SOLID'
    return ob

def new_material(id):

    mat = bpy.data.materials.get(id)

    if mat is None:
        mat = bpy.data.materials.new(name=id)

    mat.use_nodes = True

    if mat.node_tree:
        mat.node_tree.links.clear()
        mat.node_tree.nodes.clear()

    return mat

## add a material to an object, create a new diffuse material if none exists by the specified name
def add_mat(ob_name, x, face_idx = None, mat_name = None):
    if mat_name is None:
        mat_name = ob_name
        
    mat = new_material(mat_name)
    
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')

    shader = nodes.new(type='ShaderNodeBsdfDiffuse')
    if x is not None:
        nodes["Diffuse BSDF"].inputs[0].default_value = color_255(x)
    
    links.new(shader.outputs[0], output.inputs[0])

    ob = bpy.data.objects[ob_name]

    if ob.data.materials is None:
        ob.data.materials[0] = mat
    else:
        ob.data.materials.append(mat)

'''def create_obj_albedo(obj, rgb, name, weights = [0.5, 0.1, 0.1, 0.3]):
    # create material
    bu.diffuse( obj=obj, rgb=rgb, name=name )
    #print(name)
    return obj, rgb'''

def create_obj_albedo(obj, textypes, name, weights = [0.5, 0.1, 0.1, 0.3], rgb=(), texture=100, pattern = ''):
    # create material
    if texture==100: # no texture is passed
        texture = random.choices(textypes, weights = weights)[0]
    if len(rgb)<3: #no rgb is passed
        rgb = rrgb()
    if texture==0:
        bu.diffuse( obj=obj, rgb=rgb, name=name )
    elif texture==1:
        bu.voronoi( obj=obj, scale=random.uniform(1.0,10.0), name=name )
    elif texture==2:
        bu.noise( obj=obj, name=name )
    elif texture==3:
        if len(pattern)>1:
            bu.patterns(obj=obj, name=name, tex2apply=pattern)
        else:
            bu.patterns( obj=obj, name=name )
    #print(name)
    return obj, texture, rgb

def create_diffuse_albedo(obj, rgb, name):
    # create material
    bu.diffuse( obj=obj, rgb=rgb, name=name )
    return obj

def create_shape(location, size, rotation = np.array([]), shape = None):
    if rotation.size<1:
        rotation = tuple( random.uniform(0,2*math.pi) for i in range(3) )
    if shape == None:
        shape = random.choice([0, 1, 2, 3, 4])
    if shape==0:
        bpy.ops.mesh.primitive_ico_sphere_add( location=location, radius=size/2, rotation=rotation, subdivisions=4 )
        bpy.context.object.data.polygons.foreach_set('use_smooth', [True] * len(bpy.context.object.data.polygons))
    elif shape==1:
        bpy.ops.mesh.primitive_cube_add( location=location, size=size, rotation=rotation )
    elif shape==2:
        bpy.ops.mesh.primitive_ico_sphere_add( location=location, radius=size/2, rotation=rotation, subdivisions=1 )
    elif shape==3:
        bpy.ops.mesh.primitive_cylinder_add( location=location, radius=size/2, depth=size, rotation=rotation, vertices=128 )
    elif shape==4:
        bpy.ops.mesh.primitive_cone_add( location=location, radius1=size/2, radius2=0, depth=size, rotation=rotation, vertices=128 )
    return bpy.context.active_object, shape
    
def create_mesh(size, obj):
    #import pdb; pdb.set_trace()
    rotation = tuple( random.uniform(0,2*math.pi) for i in range(3) )
    shape = random.choice(objtypes)
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

