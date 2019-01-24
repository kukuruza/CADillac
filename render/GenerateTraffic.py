#!/usr/bin/env python
import sys, os, os.path as op
sys.path.insert(0, op.join(os.getenv('CITY_PATH'), 'src'))
import argparse
import json, simplejson
import logging
from pprint import pprint 
from learning.helperSetup import setupLogging
from learning.helperSetup import assertParamIsThere, setParamUnlessThere
from learning.video2dataset import make_dataset
from learning.helperDb import parseTimeString
from Cad import Cad
from Camera import Camera
from Video import Video
from traffic import TrafficModel, TrafficModelRandom
import sqlite3
from processScene import Diapason

from renderUtil import atcadillac


def generate_video_traffic (job):
  ''' Generate traffic file for the whole video.
  Args:
    in_db_file - should have all the images for which traffic is generated
    job - the same as for process_video
  '''
  assertParamIsThere  (job, 'in_db_file')
  assertParamIsThere  (job, 'out_video_dir')
  setParamUnlessThere (job, 'frame_range', '[::]')
  assertParamIsThere  (job, 'video_dir')

  video = Video(video_dir=job['video_dir'])
  camera = video.build_camera()

  assert op.exists(atcadillac(job['in_db_file'])), \
      'in db %s does not exist' % atcadillac(job['in_db_file'])
  conn_in = sqlite3.connect(atcadillac(job['in_db_file']))
  c_in = conn_in.cursor()
  c_in.execute('SELECT time FROM images')
  timestamps = c_in.fetchall()
  conn_in.close()

  cad = Cad()

  if 'speed_kph' in job:
    model = TrafficModel (camera, video, cad=cad, speed_kph=job['speed_kph'])
  elif 'num_cars' in job:
    model = TrafficModelRandom (camera, video, cad=cad, num_cars_mean=job['num_cars'])
  else: assert False

  diapason = Diapason (len(timestamps), job['frame_range'])
  
  traffic = {'in_db_file': job['in_db_file']}
  traffic['frames'] = []

  for frame_id in diapason.frame_range:
    logging.info ('generating traffic for frame %d' % frame_id)
    timestamp = timestamps[frame_id][0]
    time = parseTimeString (timestamp)
    traffic_frame = model.get_next_frame(time)
    traffic_frame['frame_id'] = frame_id  # for validating
    traffic['frames'].append(traffic_frame)

  return traffic


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--logging_level', default=20, type=int)
    parser.add_argument('--frame_range', default='[::]', 
                        help='python style ranges, e.g. "[5::2]"')
    parser.add_argument('--in_db_file', required=True)
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--traffic_file', required=True,
                        help='output .json file where to write traffic info. '
                             'Can be "traffic.json" in video output dir.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--speed_kph', type=int)
    group.add_argument('--num_cars', type=int)
    args = parser.parse_args()

    setupLogging('log/augmentation/GenerateTraffic.log', args.logging_level, 'w')

    if not op.exists(atcadillac(op.dirname(args.traffic_file))):
      os.makedirs(atcadillac(op.dirname(args.traffic_file)))
              
    job = {'frame_range':   args.frame_range,
           'in_db_file':    args.in_db_file,
           'video_dir':     args.video_dir,
           'out_video_dir': op.dirname(args.in_db_file)
    }
    if args.speed_kph is not None:
      setParamUnlessThere (job, 'speed_kph', args.speed_kph)
    elif args.num_cars is not None:
      setParamUnlessThere (job, 'num_cars', args.num_cars)
    else: assert False
    
    pprint (job)
    traffic = generate_video_traffic (job)
    print 'generated traffic for %d frames' % len(traffic['frames'])
    with open(atcadillac(args.traffic_file), 'w') as f:
      f.write(json.dumps(traffic, indent=2))

