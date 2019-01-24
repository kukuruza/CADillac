import sys, os, os.path as op
import json
import logging
import bpy

from ..render.common import *
from utils import bounds, get_x_wheels
from collectionUtilities import atcadillac, COLLECTION_WORK_DIR


def get_dims (model_id):

    # select the car as object
    obj = bpy.data.objects[model_id]
    obj.select = True

    # scale the DimsPlane to illustrate dimensions
    #plane = bpy.data.objects['DimsPlane']
    #plane.location = [0, 0, 0]

    roi = bounds(obj)
    dims = [roi.x.max-roi.x.min, roi.y.max-roi.y.min, roi.z.max-roi.z.min]
    dims = dict(zip(['x', 'y', 'z'], dims))

    x_wheels = get_x_wheels(obj)

    return {'dims': dims, 'x_wheels': x_wheels}


if __name__ == '__main__':

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    model_path = op.join(COLLECTION_WORK_DIR, 'model.json')
    model = json.load(open(model_path))

    model_id = op.basename(op.splitext(model['blend_file'])[0])
    logging.info('Processing model: %s' % model_id)

    scene_path = atcadillac('scenes/empty-import.blend')
    bpy.ops.wm.open_mainfile(filepath=scene_path)

    try:
        import_blend_car(atcadillac(model['blend_file']), model_id)
    except:
        logging.error('Could not import .blend model: %s' % atcadillac(model['blend_file']))
        model['error'] = 'blender cannot import .blend model'
        sys.exit()

    dims = get_dims (model_id)

    with open(model_path, 'w') as fid:
        fid.write(json.dumps(dims, indent=2))

