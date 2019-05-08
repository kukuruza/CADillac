# Copyright 2019 Evgeny Toropov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import bpy
import sys, os, os.path as op
sys.path.insert(0, op.join(os.getenv('CITY_PATH'), 'src'))
import json
import logging
import numpy as np
from augmentation.common import *
from learning.helperSetup import setupLogging, setParamUnlessThere

from renderUtil import atcadillac


WORK_RENDER_DIR   = atcadillac('blender/current-frame')
TRAFFIC_FILENAME  = 'traffic.json'

WORK_DIR = '%s-%d' % (WORK_RENDER_DIR, os.getppid())


def make_snapshot (render_dir, car_names, params):
    '''Set up the weather, and render vehicles into files
    Args:
      render_dir:  path to directory where to put all rendered images
      car_names:   names of car objects in the scene
      params:      dictionary with frame information
    Returns:
      nothing
    '''

    logging.info ('make_snapshot: started')

    setParamUnlessThere (params, 'scale', 1)
    setParamUnlessThere (params, 'render_individual_cars', True)
    # debug options
    setParamUnlessThere (params, 'save_blender_file', False)
    setParamUnlessThere (params, 'render_satellite', False)
    setParamUnlessThere (params, 'render_cars_as_cubes', False)

    bpy.data.worlds['World'].light_settings.environment_energy = 0.0
    bpy.data.worlds['World'].light_settings.ao_factor = 0.5
    bpy.data.objects['-Sky-sunset'].data.energy = np.random.normal(1, 0.5) #2

    params['weather'] = np.random.choice(['Sunny', 'Cloudy', 'Rainy', 'Wet'])
    set_weather (params)

    # render the image from satellite, when debuging
    if '-Satellite' in bpy.data.objects:
        bpy.data.objects['-Satellite'].hide_render = not params['render_satellite']

    # make all cars receive shadows
    logging.info ('materials: %s' % len(bpy.data.materials))
    for m in bpy.data.materials:
        m.use_transparent_shadows = True



    # create render dir
    if not op.exists(render_dir):
        os.makedirs(render_dir)

    # make all cars receive shadows
    logging.info ('materials: %s' % len(bpy.data.materials))
    for m in bpy.data.materials:
        m.use_transparent_shadows = True

    # # render all cars and shadows
    # bpy.context.scene.render.layers['RenderLayer'].use_pass_combined = True
    # bpy.context.scene.render.layers['RenderLayer'].use_pass_z = False
    # #bpy.data.objects['-Ground'].hide_render = False
    # render_scene(op.join(render_dir, 'render'))

    # # render cars depth map
    # #bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
    # bpy.context.scene.render.layers['RenderLayer'].use_pass_combined = False
    # bpy.context.scene.render.layers['RenderLayer'].use_pass_z = True
    # #bpy.data.objects['-Ground'].hide_render = True
    # render_scene(op.join(render_dir, 'depth-all'))

    # # render just the car for each car (to extract bbox)
    # if params['render_individual_cars'] and not params['render_cars_as_cubes']:
    #     # hide all cars
    #     for car_name in car_names:
    #         hide_car (car_name)
    #     # show, render, and hide each car one by one
    #     for i,car_name in enumerate(car_names):
    #         show_car (car_name)
    #         render_scene( op.join(render_dir, 'depth-car-%03d.png' % i) )
    #         hide_car (car_name)

    # # clean up
    # bpy.data.objects['-Ground'].hide_render = False
    # if not params['render_cars_as_cubes']:
    #     for car_name in car_names:
    #         show_car (car_name)

    def _rename (render_dir, from_name, to_name):
        os.rename(atcadillac(op.join(render_dir, from_name)), 
                  atcadillac(op.join(render_dir, to_name)))

    # set directional blur amount from if given
    #bpy.data.node_groups['Compositing Nodetree'].nodes['Camera-Blur'].zoom
    
    # there are two nodes -- "render" and "depth"
    # they save images in BW16 or RBG8
    # they render layers "Render" and "Depth" with "Combined" and "Z" passes.
    bpy.context.scene.node_tree.nodes['depth'].base_path = atcadillac(render_dir)
    bpy.context.scene.node_tree.nodes['render'].base_path = atcadillac(render_dir)

    # leave only shadows
    for m in bpy.data.materials:
      if m != bpy.data.materials['Material-dry-asphalt'] and \
         m != bpy.data.materials['Material-wet-asphalt']:
        m.use_only_shadow = True

    # render shadows only
    bpy.data.objects['-Ground'].hide_render = False
    bpy.ops.render.render (write_still=True, layer='Render')
    _rename (render_dir, 'render0001', 'render.png')

    # materials back to normal
    for m in bpy.data.materials:
      if m != bpy.data.materials['Material-dry-asphalt'] and \
         m != bpy.data.materials['Material-wet-asphalt']:
        m.use_only_shadow = False

    # render without ground
    bpy.data.objects['-Ground'].hide_render = True
    bpy.ops.render.render (write_still=True, layer='Render')
    _rename (render_dir, 'render0001', 'cars-only.png')

    # render depth of all cars
    bpy.ops.render.render (write_still=True, layer='Depth')
    _rename (render_dir, 'depth0001', 'depth-all.png')

    if params['render_individual_cars'] and not params['render_cars_as_cubes']:
      for car_i0, car_name0 in enumerate(car_names):

        # remove all cars from the only layer, and add car_name0 back to it
        for car_name in car_names:
            bpy.data.objects[car_name].hide_render = True
        bpy.data.objects[car_name0].hide_render = False

        # render scene
        bpy.ops.render.render (write_still=True, layer='Depth')
        _rename (render_dir, 'depth0001', 'depth-%03d.png' % car_i0)

    if params['save_blender_files']:
        bpy.ops.wm.save_as_mainfile (filepath=atcadillac(op.join(render_dir, 'render.blend')))

    # logging.info ('objects in the end of frame: %d' % len(bpy.data.objects))
    logging.info ('make_snapshot: successfully finished a frame')
    


setupLogging('log/augmentation/renderScene.log', logging.INFO, 'a')

traffic_path = op.join(WORK_DIR, TRAFFIC_FILENAME)
logging.info ('traffic_path: %s' % traffic_path)
frame_info = json.load(open(traffic_path))
setParamUnlessThere (frame_info, 'render_cars_as_cubes', False)

# place all cars
car_names = []
for i,vehicle in enumerate(frame_info['vehicles']):
    if frame_info['render_cars_as_cubes']:
        location = (vehicle['x'], vehicle['y'], 0.1)
        bpy.ops.mesh.primitive_cube_add(location=location, radius=0.3)
    else:
        collection_id = vehicle['collection_id']
        model_id = vehicle['model_id']
        blend_path = atcadillac(op.join('CAD', collection_id, 'blend', '%s.blend' % model_id))
        car_name = 'car_%i' % i
        car_names.append(car_name)
        import_blend_car (blend_path, model_id, car_name)
        position_car (car_name, x=vehicle['x'], y=vehicle['y'], azimuth=vehicle['azimuth'])
        scale = frame_info['scale']
        bpy.ops.transform.resize (value=(scale, scale, scale))

make_snapshot (WORK_DIR, car_names, frame_info)
