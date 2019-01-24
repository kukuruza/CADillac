import bpy
import sys, os, os.path as op
import json
from math import cos, sin, pi, sqrt
import numpy as np
import logging
from numpy.random import normal, uniform
from mathutils import Color, Euler


def dump(obj):
   '''Helper function to output all properties of an object'''
   for attr in dir(obj):
       if hasattr( obj, attr ):
           print( "obj.%s = %s" % (attr, getattr(obj, attr)))



def render_scene (filepath):
    if op.exists(filepath): 
        os.remove(filepath)
    bpy.data.scenes['Scene'].render.filepath = filepath
    bpy.ops.render.render (write_still=True) 



def delete_car (car_name):
    assert car_name in bpy.data.objects, '%s' % car_name

    # deselect all
    bpy.ops.object.select_all(action='DESELECT')  
    
    bpy.data.objects[car_name].select = True
    bpy.ops.object.delete()



def import_car_obj (obj_path, car_group_name):
    assert car_group_name not in bpy.data.groups, '%s' % car_group_name

    car_group = bpy.data.groups.new(car_group_name)

    bpy.ops.import_scene.obj (filepath=obj_path)

    # add all new objects (they are all selected now) to the group
    for obj in bpy.context.selected_objects:
        bpy.context.scene.objects.active = obj
        bpy.ops.object.group_link (group=car_group_name)

    assert car_group_name in bpy.data.groups
    logging.debug ('in group "%s" there are %d objects' % 
        (car_group_name, len(bpy.data.groups[car_group_name].objects)))


def import_blend_car (blend_path, model_id, car_name=None):
    '''Import model_id object from blend_path .blend file, and rename it to car_name
    '''
    # append object from .blend file
    assert op.exists(blend_path)
    with bpy.data.libraries.load(blend_path, link=False) as (data_src, data_dst):
        data_dst.objects = [model_id]

    # link object to current scene
    obj = data_dst.objects[0]
    assert obj is not None
    bpy.context.scene.objects.link(obj)

    # raname
    if car_name is None: car_name = model_id
    obj.name = car_name
    logging.debug ('model_id %s imported' % model_id)



def join_car_meshes (model_id):
    ''' Join all meshes in a model_id group into a single object. Keep group
    Return:
        active object (joined model)
    '''
    for obj in bpy.data.groups[model_id].objects:
        bpy.data.objects[obj.name].select = True
    bpy.context.scene.objects.active = bpy.data.groups[model_id].objects[0]
    bpy.ops.object.join()
    bpy.context.scene.objects.active.name = model_id
    return bpy.context.scene.objects.active



def hide_car (car_name):
    '''Tags car object invisible'''
    assert car_name in bpy.data.objects

    bpy.data.objects[car_name].hide = True
    bpy.data.objects[car_name].hide_render = True


def show_car (car_name):
    '''Tags car object visible'''
    assert car_name in bpy.data.objects

    bpy.data.objects[car_name].hide = False
    bpy.data.objects[car_name].hide_render = False





def set_rainy ():
    
    set_wet()

    # turn on mist
    bpy.data.worlds['World'].mist_settings.use_mist = True
    bpy.data.worlds['World'].mist_settings.intensity = 0.1 #normal(0.2, 0.07)
    bpy.data.worlds['World'].mist_settings.depth = normal(150., 30.)
    bpy.data.worlds['World'].mist_settings.falloff = 'INVERSE_QUADRATIC'
    logging.info ('rainy weather set mist to %0.1f' % 
      bpy.data.worlds['World'].mist_settings.depth)


def set_wet ():

    # trun off sun
    sun = bpy.data.objects['-Sun']
    sun.hide_render = True
    sun.hide = True

    # pick the material
    mat = bpy.data.materials['Material-wet-asphalt']
    ground = bpy.data.objects['-Ground']
    if len(ground.data.materials):
        ground.data.materials[0] = mat  # assign to 1st material slot
    else:
        ground.data.materials.append(mat)  # no slots

    bpy.data.objects['-Sky-sunset'].data.energy = 1.0

    bpy.data.worlds['World'].light_settings.environment_energy = 0.5

    bpy.data.worlds['World'].mist_settings.use_mist = False


def set_sunny ():

    mat = bpy.data.materials['Material-dry-asphalt']
    ground = bpy.data.objects['-Ground']
    if len(ground.data.materials):
        ground.data.materials[0] = mat  # assign to 1st material slot
    else:
        ground.data.materials.append(mat)  # no slots

    # adjust sun
    sun = bpy.data.objects['-Sun']
    sun.hide_render = False
    sun.hide = False
    sun.data.energy = normal(3, 1.0)   # was 4.5
    sun.data.color = (1.0000, 0.9163, 0.6905)

    bpy.data.objects['-Sky-sunset'].data.energy = 1.0

    bpy.data.worlds['World'].light_settings.environment_energy = normal(1.0, 0.2)

    bpy.data.worlds['World'].mist_settings.use_mist = False


def set_cloudy ():

    mat = bpy.data.materials['Material-dry-asphalt']
    ground = bpy.data.objects['-Ground']
    if len(ground.data.materials):
        ground.data.materials[0] = mat  # assign to 1st material slot
    else:
        ground.data.materials.append(mat)  # no slots

    # trun off sun
    sun = bpy.data.objects['-Sun']
    sun.hide_render = True
    sun.hide = True

    bpy.data.objects['-Sky-sunset'].data.energy = 1.0

    bpy.data.worlds['World'].light_settings.environment_energy = normal(2, 0.5)

    bpy.data.worlds['World'].mist_settings.use_mist = False


def set_sun_angle (azimuth, altitude):
    '''
    Args:
      altitude: angle from surface, in degrees
      azimuth:  angle from the north, in degrees. On the east azimuth equals +90
    '''
    # note: azimuth lamp direction is the opposite to sun position
    yaw   = - (azimuth - 90) * pi / 180
    pitch = (90 - altitude) * pi / 180

    # set orientation
    sun = bpy.data.objects['-Sun']
    sun.rotation_euler = Euler((0, pitch, yaw), 'ZXY')

    # two opposite colors -- noon and sunset
    c_noon   = np.asarray([0.125, 0.151, 1])
    c_sunset = np.asarray([0, 0.274, 1])
    # get the mix between them according to the time of the day
    k = pitch / (pi/2)  # [0, 1], 0 -- noon, 1 - sunset
    c = Color()
    c.hsv = tuple(c_noon * (1 - k) + c_sunset * k)
    print ('set_sun_angle: pitch=%f, k=%f, c=(%.3f, %.3f, %.3f)' % (pitch, k, c[0], c[1], c[2]))
    sun.data.color = c



def set_weather (params):
    '''Set sun and weather conditions
    '''
    weather = params['weather']

    if weather == 'Rainy': 
        logging.info ('setting rainy weather')
        set_rainy()
    elif weather == 'Wet':
        logging.info ('setting wet weather')
        set_wet()
    elif weather == 'Cloudy':
        logging.info ('setting cloudy weather')
        set_cloudy()
    elif weather == 'Sunny': 
        alt = params['sun_altitude']
        azi = params['sun_azimuth']
        logging.info ('setting sunny weather with azimuth,altitude = %f,%f' % (azi, alt))
        set_sunny()
        set_sun_angle(azi, alt)
    else:
        raise Exception ('Invalid weather param: %s' % str(weather))



def position_car (car_name, x, y, azimuth):
    '''Put the car to a certain position on the ground plane
    Args:
      car_name:        name of the blender model
      x, y:            target position in the blender x,y coordinate frame
      azimuth:         angle in degrees, 0 is North (y-axis) and 90 deg. is East
    '''
    # TODO: now assumes object is at the origin.
    #       instead of transform, assign coords and rotation

    assert car_name in bpy.data.objects

    # select only car
    bpy.ops.object.select_all(action='DESELECT')  
    bpy.data.objects[car_name].select = True

    bpy.ops.transform.translate (value=(x, y, 0))
    bpy.ops.transform.rotate (value=(90 - azimuth) * pi / 180, axis=(0,0,1))
