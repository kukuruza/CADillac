import sys, os, os.path as op
import json
import logging
import bpy

from ..render.common import *
from utils import bounds
from collectionUtilities import COLLECTION_WORK_DIR, atcadillac


def render_example (model_id):

    # select the car as object
    obj = bpy.data.objects[model_id]
    obj.select = True

    # scale the DimsPlane to illustrate dimensions
    plane = bpy.data.objects['DimsPlane']
    plane.location = [0, 0, 0]

    roi = bounds(obj)
    dims = [roi.x.max-roi.x.min, roi.y.max-roi.y.min, roi.z.max-roi.z.min]
    dims = dict(zip(['x', 'y', 'z'], dims))

    plane.scale.x = dims['x'] * 0.5
    plane.scale.y = dims['y'] * 0.5

    # save a rendered example
    example_file = model['example_file']
    if not op.exists(atcadillac(op.dirname(example_file))): 
        os.makedirs(atcadillac(op.dirname(example_file)))
    logging.info ('writing example to %s' % example_file)
    render_scene(atcadillac(example_file))



if __name__ == '__main__':

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    model_path = op.join(COLLECTION_WORK_DIR, 'model.json')
    model = json.load(open(model_path))

    model_id = model['model_id']
    logging.info ('processing model: %s' % model_id)

    scene_path = atcadillac('scenes/empty-import.blend')
    bpy.ops.wm.open_mainfile(filepath=scene_path)

    try:
        import_blend_car (atcadillac(model['blend_file']), model_id)
    except:
        logging.error('could not import .blend model: %s' % atcadillac(model['blend_file']))
        model['error'] = 'blender cannot import .blend model'
        sys.exit()

    render_example (model_id)
