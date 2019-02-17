import bpy
import sys, os, os.path as op
import json
import logging
from math import cos, sin, pi, sqrt, ceil
import numpy as np
from glob import glob
from random import choice
from numpy.random import normal, uniform
from mathutils import Color, Euler

sys.path.insert(0, op.dirname(op.dirname(os.path.realpath(__file__)))) # = '..'
from render.common import *
from cads.collectionUtilities import getBlendPath
from render.renderUtil import atcadillac


COLLECTIONS_DIR  = atcadillac('CAD')
ROAD_TEXTURE_DIR = atcadillac('resources/textures/road')
BLDG_TEXTURE_DIR = atcadillac('resources/textures/buildings')
WORK_PATCHES_DIR = '/tmp/blender/current-patch'
JOB_INFO_NAME    = 'job_info.json'
OUT_INFO_NAME    = 'out_info.json'

WORK_DIR = '%s-%d' % (WORK_PATCHES_DIR, os.getppid())


SCALE_FACTOR = 3   # how far a camera is from the origin
RENDER_WIDTH  = 600
RENDER_HEIGHT = 600

# sampling weather and camera position
SUN_ALTITUDE_MIN  = 20
SUN_ALTITUDE_MAX  = 70


def choose_params(azimuth_low, azimuth_high, pitch_low, pitch_high):
    ''' Pick some random parameters, adjust lighting, and finally render a frame. '''

    # pick random weather
    params = {}
    params['sun_azimuth']  = uniform(low=0, high=360)
    params['sun_altitude'] = uniform(low=SUN_ALTITUDE_MIN, high=SUN_ALTITUDE_MAX)
    params['weather'] = choice(['Rainy', 'Cloudy', 'Sunny', 'Wet'])
    set_weather (params)

    # pick random camera angle and distance
    params['azimuth'] = uniform (low=azimuth_low, high=azimuth_high)
    params['altitude'] = uniform (low=pitch_low, high=pitch_high)
    logging.info ('prepare_photo: azimuth (deg): %.1f, altitude: %.1f (deg)' %
        (params['azimuth'], params['altitude']))
   
    # assign a random texture from the directory
    road_texture_path = choice(glob(op.join(ROAD_TEXTURE_DIR, '*.jpg')))
    logging.info ('road_texture_path: %s' % road_texture_path)
    params['road_texture_path'] = road_texture_path
    # pick a random road width
    params['road_width'] = normal(15, 5)

    # assign a random texture from the directory
    buidling_texture_path = choice(glob(op.join(BLDG_TEXTURE_DIR, '*.jpg')))
    logging.info ('buidling_texture_path: %s' % buidling_texture_path)
    params['buidling_texture_path'] = buidling_texture_path
    # pick a random height dim
    params['building.dimensions.z'] = normal(20, 5)
    # move randomly along a X and Z axes
    params['building.location.x'] = normal(0, 5)
    params['building.location.z'] = uniform(-5, 0)

    return params


def make_snapshot (car_sz, render_dir, car_names, params):
    '''Set up the weather, and render vehicles into files
    Args:
      render_dir:  path to directory where to put all rendered images
      car_names:   names of car objects in the scene
      params:      dictionary with frame information
    Returns:
      nothing
    '''
    logging.info ('make_snapshot: started')

    # compute camera position
    azimuth_rad = params['azimuth'] * pi / 180
    altitude_rad = params['altitude'] * pi / 180
    dist  = car_sz * SCALE_FACTOR
    x = dist * cos(azimuth_rad) * cos(altitude_rad)
    y = dist * sin(azimuth_rad) * cos(altitude_rad)
    z = dist * sin(altitude_rad)
    bpy.data.objects['-Camera'].location = (x,y,z)

    # set up lighting
    bpy.data.objects['-Sky-sunset'].location = (-x,-y,10)
    bpy.data.objects['-Sky-sunset'].rotation_euler = (60*pi/180, 0, azimuth_rad-pi/2)

    # set up road
    bpy.data.images['ground'].filepath = params['road_texture_path']
    bpy.data.objects['-Ground'].dimensions.x = params['road_width']

    # set up building
    bpy.data.images['building'].filepath = params['buidling_texture_path']
    bpy.data.objects['-Building'].dimensions.z = params['building.dimensions.z']
    bpy.data.objects['-Building'].location.x = params['building.location.x']
    bpy.data.objects['-Building'].location.z = params['building.location.z']
    # put the building at the edge of the road, opposite to the camera
    bpy.data.objects['-Building'].location.y = params['road_width'] / 2 * (1 if y < 0 else -1)
    
    # nodes to change output paths
    bpy.context.scene.node_tree.nodes['render'].base_path = atcadillac(render_dir)
    bpy.context.scene.node_tree.nodes['depth-all'].base_path = atcadillac(render_dir)
    bpy.context.scene.node_tree.nodes['depth-car'].base_path = atcadillac(render_dir)
    bpy.context.scene.node_tree.nodes['depth-road'].base_path = atcadillac(render_dir)
    bpy.context.scene.node_tree.nodes['depth-building'].base_path = atcadillac(render_dir)

    # make all cars receive shadows
    logging.info ('materials: %s' % len(bpy.data.materials))
    for m in bpy.data.materials:
        m.use_transparent_shadows = True

    # add cars to Cars and Depth-all layers and the main car to Depth-car layer
    for car_name in car_names:
        for layer_id in range(5):
            bpy.data.objects[car_name].layers[layer_id] = False
    for car_name in car_names:
        bpy.data.objects[car_name].layers[0] = True
        bpy.data.objects[car_name].layers[1] = True
    bpy.data.objects[car_names[0]].layers[2] = True

    # Render scene.
    # I get the result render, but the command throws:
    #   "Error: Render error (No such file or directory) cannot save: ''"
    #   For now, put it into try-except block.
    try:
        bpy.ops.render.render (write_still=True, layer='Cars')
    except:
        pass

    # change the names of output png files
    for layer_name in ['render', 'depth-all', 'depth-car', 'depth-road', 'depth-building']:
        os.rename(atcadillac(op.join(render_dir, '%s0001' % layer_name)), 
                  atcadillac(op.join(render_dir, '%s.png' % layer_name)))

    ### aftermath
    
    for car_name in car_names:
        show_car (car_name)

    logging.info ('make_snapshot: successfully finished a frame')
    


def photo_session (job):
    '''Take pictures of a scene from different random angles, 
      given some cars placed and fixed in the scene.
    '''
    num_per_session = job['num_per_session']
    vehicles        = job['vehicles']

    azimuth_low = job['azimuth_low']
    azimuth_high = job['azimuth_high']
    pitch_low = job['pitch_low']
    pitch_high = job['pitch_high']

    # open the blender file
    bpy.context.user_preferences.filepaths.use_relative_paths = False
    scene_path = op.join(op.dirname(os.path.realpath(__file__)), 'resources/photo-session.blend')
    logging.info('Looking for photo-session.blend at %s' % scene_path)
    bpy.ops.wm.open_mainfile (filepath=scene_path)

    render_dir = op.join(WORK_DIR, '%06d' % 0)
    if not op.exists(atcadillac(render_dir)):
        os.makedirs(atcadillac(render_dir))
    if job['save_blender']:
        bpy.ops.wm.save_as_mainfile (filepath=atcadillac(op.join(render_dir, 'out1.blend')))

    if 'render_width' not in job: job['render_width'] = RENDER_WIDTH
    if 'render_height' not in job: job['render_height'] = RENDER_HEIGHT
    bpy.context.scene.render.resolution_x = job['render_width']
    bpy.context.scene.render.resolution_y = job['render_height']

    car_names = []
    for i,vehicle in enumerate(vehicles):
        blend_path = getBlendPath(vehicle['collection_id'], vehicle['model_id'])

        assert op.exists(blend_path), 'blend path does not exist' % blend_path
        # if 'dims' not in vehicle or not op.exists(blend_path):
        #     logging.error ('dims or blend_path does not exist. Skip.')
        #     continue

        car_name = 'car-%d' % i
        car_names.append(car_name)
        import_blend_car (blend_path, vehicle['model_id'], car_name)
        position_car (car_name, vehicles[i]['x'], vehicles[i]['y'], vehicles[i]['azimuth'])

    # take snapshots from different angles
    dims = vehicles[0]['dims']  # dict with 'x', 'y', 'z' in meters
    car_sz = sqrt(dims['x']**2 + dims['y']**2 + dims['z']**2)
    for i in range(num_per_session):
        render_dir = op.join(WORK_DIR, '%06d' % i)
        # create render dir
        if not op.exists(atcadillac(render_dir)):
            os.makedirs(atcadillac(render_dir))

        use_90turn = job['use_90turn'] if 'use_90turn' in job else False
        logging.info ('use_90turn %s' % str(use_90turn))
        if use_90turn and i % 2 == 1:
          logging.info ('using use_90turn')
          params['azimuth'] = (params['azimuth'] + 90) % 360
        else:
          logging.info ('PREPARED: using azimuth: %.1f - %.1f, pitch: %.1f - %.1f' % 
            (azimuth_low, azimuth_high, pitch_low, pitch_high))
          params = choose_params(azimuth_low, azimuth_high, pitch_low, pitch_high)

        if job['save_blender']:
            bpy.ops.wm.save_as_mainfile (filepath=atcadillac(op.join(render_dir, 'out2.blend')))

        make_snapshot (car_sz, render_dir, car_names, params)

        ### write down some labelling info
        out_path = atcadillac(op.join(render_dir, OUT_INFO_NAME))
        with open(out_path, 'w') as f:
            f.write(json.dumps({'azimuth':      (180 - params['azimuth']) % 360, # match KITTI
                                'altitude':     params['altitude'],
                                'model_id':     vehicles[0]['model_id'],
                                'color':        vehicles[0]['color']
                                }, indent=4))

job = json.load(open( op.join(WORK_DIR, JOB_INFO_NAME) ))
logging.basicConfig(level=job['logging'], stream=sys.stderr, 
    format='%(levelname)s:photosession: %(message)s')

photo_session (job)

