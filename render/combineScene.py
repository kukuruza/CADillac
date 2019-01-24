import sys, os, os.path as op
sys.path.insert(0, op.join(os.getenv('CITY_PATH'), 'src'))
import json
import logging
import bpy
import numpy as np
from learning.helperSetup import setupLogging, setParamUnlessThere

from renderUtil import atcadillac

''' Make all frame postprocessing and combination in RENDER_DIR '''

WORK_RENDER_DIR = atcadillac('blender/current-frame')
BACKGROUND_FILENAME = 'background.png'
NORMAL_FILENAME     = 'render.png'
CARSONLY_FILENAME   = 'cars-only.png'
COMBINED_FILENAME   = 'out.png'
#CORRECTION_FILENAME = 'color-correction.json'

WORK_DIR = '%s-%d' % (WORK_RENDER_DIR, os.getppid())
WORK_DIR_SUFFIX = '-%d' % os.getppid()

#correction_path = op.join(WORK_DIR, CORRECTION_FILENAME)

image_node = bpy.context.scene.node_tree.nodes['Image-Background'].image
image_node.filepath = op.join(WORK_DIR, BACKGROUND_FILENAME)

image_node = bpy.context.scene.node_tree.nodes['Image-Cars-Only'].image
image_node.filepath = op.join(WORK_DIR, CARSONLY_FILENAME)

image_node = bpy.context.scene.node_tree.nodes['Image-Normal'].image
image_node.filepath = op.join(WORK_DIR, NORMAL_FILENAME)


bpy.context.scene.node_tree.nodes['Hue-Saturation-Compensation'].color_saturation *= np.random.normal(1, 0.2)
bpy.context.scene.node_tree.nodes['Hue-Saturation-Compensation'].color_value *= np.random.normal(1, 0.2)

# render and save
# TODO: should delete bpy.data.scenes['Scene'].render.filepath ??
bpy.data.scenes['Scene'].render.filepath = op.join(WORK_DIR, COMBINED_FILENAME)
bpy.ops.render.render (write_still=True) 

bpy.ops.wm.save_as_mainfile (filepath=op.join(WORK_DIR, 'combine.blend'))

