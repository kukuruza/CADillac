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


import sys, os, os.path as op
import json
import logging
import bpy

from ..render.common import *
from collectionUtilities import COLLECTION_WORK_DIR, atcadillac


def scaleModel (model_id, scale):
    obj = bpy.data.objects[model_id]
    obj.select = True
    bpy.ops.transform.resize (value=(scale, scale, scale))


if __name__ == '__main__':

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    model_path = op.join(COLLECTION_WORK_DIR, 'model.json')
    model = json.load(open(model_path))

    scale = model['scale']
    logging.info('Will scale with f=%f' % scale)

    dry_run = model['dry_run']
    logging.info('Dry run mode is %s.' % ('on' if dry_run else 'off'))

    model_id = op.basename(op.splitext(model['blend_file'])[0])
    logging.info('Processing model: %s' % model_id)

    scene_path = atcadillac('scenes/empty-import.blend')
    bpy.ops.wm.open_mainfile(filepath=atcadillac(model['blend_file']))

    logging.info('Import succeeded.')
    scaleModel (model_id, scale)
    status = 'ok'

    if not dry_run:
      bpy.ops.wm.save_as_mainfile(filepath=atcadillac(model['blend_file']))

    with open(model_path, 'w') as fid:
        fid.write(json.dumps({'status': status}, indent=2))
