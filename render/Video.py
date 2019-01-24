import sys, os, os.path as op
from glob import glob
from datetime import datetime
import logging
import json
import cv2
import argparse
from Camera import Camera
from learning.helperSetup import setupLogging

from renderUtil import atcadillac

VIDEO_DIR_REGEX     = r'^[A-Z][a-z].*[0-9]{2}-[0-9]{2}h$'
VIDEO_DIR_STRPTIME  = '%b%d-%Hh'
TIME_FORMAT         = '%Y-%m-%d %H:%M:%S.%f'

class Video:

    def __init__ (self, video_dir=None, video_info=None):

        if video_dir:
            video_name = op.basename(video_dir)
            video_path = atcadillac(op.join(video_dir, '%s.json' % video_name))
            assert op.exists(video_path), video_path
            logging.info ('Video: loading info from: %s' % video_path)
            video_info = json.load(open(video_path))
        elif video_info:
            assert 'video_dir' in video_info
            video_dir = video_info['video_dir']
            assert op.exists(atcadillac(video_dir)), video_dir
            video_name = op.basename(video_dir)
        else:
            raise Exception ('pass video_info or video_dir')
        logging.info ('Video: parse info for: %s' % video_dir)

        self.info = video_info
        self.info['video_name'] = video_name

        if 'example_frame_name' in video_info:
            logging.info ('- found example_frame_name: %s' % example_frame_name)
            self.example_frame = cv2.imread(op.join(video_dir, example_frame_name))
            assert self.example_frame is not None
        else:
            # trying possible paths, and take the first to match
            example_frame_paths = glob (atcadillac(op.join(video_dir, 'frame*.png')))
            if len(example_frame_paths) > 0:
                logging.info ('- deduced example_frame: %s' % example_frame_paths[0])
                self.example_frame = cv2.imread(example_frame_paths[0])
                self.info['example_frame_name'] = op.basename(example_frame_paths[0])
                assert self.example_frame is not None
            else:
                logging.warning ('- no example_frame for %s' % video_dir)
                self.example_frame = None
                self.info['example_frame_name'] = None

        if 'example_background_name' in video_info:
            example_background_name = video_info['example_background_name']
            logging.info ('- found example_background_name: %s' % example_background_name)
            example_background_path = atcadillac(op.join(video_dir, example_background_name))
            self.example_background = cv2.imread(example_background_path)
            assert self.example_background is not None
        else:
            # trying possible paths
            example_back_paths = glob (atcadillac(op.join(video_dir, 'background*.png')))
            if len(example_back_paths) > 0:
                logging.info ('- deduced example_background: %s' % example_back_paths[0])
                self.example_background = cv2.imread(example_back_paths[0])
                assert self.example_background is not None
            else:
                logging.warning ('- no example_background for %s' % video_dir)
                self.example_background = None

        if 'start_timestamp' in video_info:
            start_timestamp = video_info['start_timestamp']
            logging.info ('- found start_timestamp: %s' % start_timestamp)
            self.start_time = datetime.strptime(start_timestamp, TIME_FORMAT)
        else:
            # deduce from the name of the file
            self.start_time = datetime.strptime(video_name, VIDEO_DIR_STRPTIME)
            logging.info ('- deduced start_time: %s' % self.start_time.strftime(TIME_FORMAT))

        if 'frame_range' in video_info:
            self.info['frame_range'] = video_info['frame_range']
            logging.info ('- found frame_range: %s' % video_info['frame_range'])
        else:
            self.info['frame_range'] = ':'

        if 'render_blend_file' in video_info:
            self.render_blend_file = video_info['render_blend_file']
            logging.info ('- found render_blend_file: %s' % self.render_blend_file)
            assert op.exists(atcadillac(self.render_blend_file))
        elif op.exists(atcadillac(op.join(video_dir, 'render.blend'))):
            # if found the default name in the video folder
            self.render_blend_file = op.join(video_dir, 'render.blend')
            logging.info ('- found render_blend_file in video dir: %s' % self.render_blend_file)
        elif op.exists(atcadillac(op.join(video_dir, 'render-generated.blend'))):
            # if found the default name in the video folder
            self.render_blend_file = op.join(video_dir, 'render-generated.blend')
            logging.warning ('- using generated render_blend_file: %s' % self.render_blend_file)
        else:
            logging.warning ('- could not figure out render_blend_file')

        if 'combine_blend_file' in video_info:
            self.combine_blend_file = video_info['combine_blend_file']
            logging.info ('- found combine_blend_file: %s' % self.combine_blend_file)
            op.exists(atcadillac(self.combine_blend_file))
        elif op.exists(atcadillac(op.join(video_dir, 'combine.blend'))):
            # if found the default name in the video folder
            self.combine_blend_file = op.join(video_dir, 'combine.blend')
            logging.info ('- found combine_blend_file in video dir: %s' % self.combine_blend_file)
        elif op.exists(atcadillac(op.join(video_dir, 'combine-generated.blend'))):
            # if found the default name in the video folder
            self.combine_blend_file = op.join(video_dir, 'combine-generated.blend')
            logging.warning ('- using generated combine_blend_file: %s' % self.combine_blend_file)
        else:
            logging.warning ('- could not figure out combine_blend_file')

        if 'camera_dir' in video_info:
            self.camera_dir = video_info['camera_dir']
            logging.info ('- found camera_dir: %s' % self.camera_dir)
        else:
            # deduce from the name of the file
            self.camera_dir = op.dirname(video_dir)
            logging.info ('- deduced camera_dir: %s' % self.camera_dir)
        assert op.exists(atcadillac(self.camera_dir)), atcadillac(self.camera_dir)

        if 'pose_id' in video_info:
            self.pose_id = int(video_info['pose_id'])
            logging.info ('- found pose_id: %d' % self.pose_id)
        else:
            self.pose_id = 0
            logging.info ('- take default pose_id = 0')


    def __getitem__(self, key):
        return self.info[key]

    def __contains__(self, key):
        return True if key in self.info else False


    def build_camera (self):
        return Camera (camera_dir=self.camera_dir, pose_id=self.pose_id)

         

if __name__ == "__main__":

    setupLogging ('log/augmentation/Video.log', logging.DEBUG, 'w')

    video = Video(video_dir='augmentation/scenes/cam578/Mar15-10h')
    camera = video.build_camera()

    #video_file = 'augmentation/scenes/cam578/Mar15-10h/Mar15-10h.json'
    #video_info = json.load(open(atcadillac(video_file)))
    #video_info['video_dir'] = 'augmentation/scenes/cam578/Mar15-10h'
    #video.load(video_info=video_info)

    #assert video.example_background is not None
    #cv2.imshow('test', video.example_background)
    #cv2.waitKey(-1)


