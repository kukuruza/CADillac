#!/usr/bin/env python
import sys, os, os.path as op
sys.path.insert(0, op.join(os.getenv('CITY_PATH'), 'src'))
import argparse
import json
import logging
from processScene import process_video
from learning.helperSetup import setupLogging, setParamUnlessThere



def make_job(args):
  job = {}
  job['frame_range'] = args.frame_range
  if args.timeout:
      job['timeout'] = args.timeout
  job['save_blender_files'] = args.save_blender_files
  job['no_annotations'] = args.no_annotations
  job['traffic_file'] = args.traffic_file
  job['video_dir'] = args.video_info_dir
  setParamUnlessThere (job, 'out_video_dir', op.dirname(args.traffic_file))
  return job


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_blender_files', action='store_true')
    parser.add_argument('--logging_level', default=20, type=int)
    parser.add_argument('--timeout', type=int, 
                        help='maximum running time, in munutes')
    parser.add_argument('--no_annotations', action='store_true',
                        help='will speed up rendering since individual cars wont be rendered')
    parser.add_argument('--frame_range', default='[::]', 
                        help='python style ranges, e.g. "[5::2]"')
    parser.add_argument('--video_info_dir', required=True)
    parser.add_argument('--traffic_file', required=True)
    args = parser.parse_args()

    setupLogging('log/augmentation/ProcessVideo.log', args.logging_level, 'w')

    job = make_job(args)
    process_video(job)
