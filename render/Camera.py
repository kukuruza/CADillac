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
import logging
import simplejson as json
from db.lib.helperSetup import setupLogging

from renderUtil import atcadillac


class Camera:

    def __init__ (self, camera_dir=None, info=None, pose_id=0):

        # get camera_info (dict) and camera_name
        if camera_dir:
            self.camera_name = op.basename(camera_dir)
            camera_path = atcadillac(op.join(camera_dir, '%s.json' % self.camera_name))
            assert op.exists(camera_path), camera_path
            logging.info ('Camera: loading info from: %s' % camera_path)
            self.info = json.load(open(camera_path))
            self.info['camera_dir'] = camera_dir
        elif info:
            assert 'camera_dir' in info
            self.info = info
            self.info['camera_dir'] = info['camera_dir']
            self.camera_name = op.dirname(info['camera_dir'])
        else:
            raise Exception ('pass camera_info or camera_dir')
        logging.info ('Camera: parse info for: %s' % self.camera_name)

        # read the proper camera_pose
        assert 'camera_poses' in self.info
        assert pose_id < len(self.info['camera_poses'])
        logging.info ('- using camera_pose %d' % pose_id)
        self.info.update(self.info['camera_poses'][pose_id])
        assert 'map_id' in self.info
        del self.info['camera_poses']

        # the default scene geometry file
        if 'geometry_blend_name' not in self.info: 
            self.info['geometry_blend_name'] = 'geometry.blend'

        # the default scale
        if 'scale' not in self.info: 
            self.info['scale'] = 1

        # read the proper google_map
        assert 'google_maps' in self.info
        map_id = self.info['map_id']
        assert map_id < len(self.info['google_maps'])
        logging.info ('- using google_maps %d' % map_id)
        self.info.update(self.info['google_maps'][map_id])
        del self.info['google_maps']

        logging.debug (json.dumps(self.info, indent = 4))


    def __getitem__(self, key):
        return self.info[key]

    def __contains__(self, key):
        return True if key in self.info else False
        


if __name__ == "__main__":

    setupLogging ('log/augmentation/Video.log', logging.DEBUG, 'w')

    camera = Camera(camera_dir='augmentation/scenes/cam717')
