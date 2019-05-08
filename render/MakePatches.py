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
sys.path.insert(0, op.join(os.getenv('SHUFFLER_PATH'), 'lib'))
import json
import logging
import subprocess
import multiprocessing
import traceback
import shutil
import argparse
import progressbar
import sqlite3
import numpy as np
import cv2
from numpy.random import normal, uniform
from random import shuffle, sample, choice
from glob import glob
from math import ceil, pi
from pprint import pprint, pformat
from imageio import imread, imwrite

if os.getenv('SHUFFLER_PATH') is None:
  raise Exception('Environmental variable SHUFFLER_PATH is not defined.')
from interfaceWriter import DatasetWriter

from renderUtil import atcadillac

WORK_PATCHES_DIR = atcadillac('/tmp/blender/current-patch')
JOB_INFO_NAME    = 'job_info.json'
OUT_INFO_NAME    = 'out_info.json'

FNULL = open('/dev/null', 'w')

# placing other cars
PROB_SAME_LANE    = 0.3
SIGMA_AZIMUTH     = 5
SIGMA_SAME        = 0.3
SIGMA_SIDE        = 0.1
MEAN_SAME         = 1.5
MEAN_SIDE         = 1.5


def mask2bbox (mask):
    '''Extract a single (if any) bounding box from the image
    Args:
      mask:  boolean mask of the car
    Returns:
      bbox:  (x1, y1, width, height)
    '''
    assert mask is not None
    assert len(mask.shape) == 2, 'mask.shape: %s' % str(mask.shape)

    # keep only vehicles with resonable bboxes
    if np.count_nonzero(mask) == 0:
        return None

    # get bbox
    nnz_indices = np.argwhere(mask)
    if len(nnz_indices) == 0:
      return None
    (y1, x1), (y2, x2) = nnz_indices.min(0), nnz_indices.max(0) + 1 
    (height, width) = y2 - y1, x2 - x1
    return (x1, y1, width, height)


def pick_spot_for_a_vehicle (dims0, model):
  '''Given car dimensions, randomly pick a spot around the main car

  Args:
    dims0:     dict with fields 'x' and 'y' for the main model
    model:     a dict with field 'dims'
  Returns:
    vehicle:   same as model, but with x,y,azimuth fields
  '''
  # in the same lane or on the lanes on the sides
  is_in_same_lane = (uniform() < PROB_SAME_LANE)

  # define probabilities for other vehicles
  if is_in_same_lane:
    x = normal(MEAN_SAME, SIGMA_SAME) * choice([-1,1])
    y = normal(0, SIGMA_SAME)
  else: 
    x = normal(0, 1.5)
    y = normal(MEAN_SIDE, SIGMA_SIDE) * choice([-1,1])

  # normalize to our car size
  x *= (dims0['x'] + model['dims']['x']) / 2
  y *= (dims0['y'] + model['dims']['y']) / 2
  azimuth = normal(0, SIGMA_AZIMUTH) + 90
  vehicle = model
  vehicle['x'] = x
  vehicle['y'] = y
  vehicle['azimuth'] = azimuth
  return vehicle


def place_occluding_vehicles (vehicle0, other_models):
  '''Distributes existing models across the scene.
  Vehicle[0] is the main photo-session character. It is in the center.

  Args:
    vehicles0:       main model dict with x,y,azimuth
    other_models:    dicts without x,y,azimuth
  Returns:
    other_vehicles:  same as other_models, but with x,y,azimuth fields
  '''
  INTERCAR_DIST_COEFF = 0.2

  vehicles = []
  for i,model in enumerate(other_models):
    logging.debug ('place_occluding_vehicles: try %d on model_id %s' 
                    % (i, model['model_id']))

    # pick its location and azimuth
    assert 'dims' in vehicle0, '%s' % json.dumps(vehicle0, indent=4)
    vehicle = pick_spot_for_a_vehicle(vehicle0['dims'], model)

    # find if it intersects with anything (cars are almost parallel)
    x1     = vehicle['x']
    y1     = vehicle['y']
    dim_x1 = vehicle['dims']['x']
    dim_y1 = vehicle['dims']['y']
    does_intersect = False
    # compare to all previous vehicles (O(N^2) haha)
    for existing_vehicle in vehicles:
      x2 = existing_vehicle['x']
      y2 = existing_vehicle['y']
      dim_x2 = existing_vehicle['dims']['x']
      dim_y2 = existing_vehicle['dims']['y']
      if (abs(x1-x2) < (dim_x1+dim_x2)/2 * (1+INTERCAR_DIST_COEFF) and 
          abs(y1-y2) < (dim_y1+dim_y2)/2 * (1+INTERCAR_DIST_COEFF)):
        logging.debug ('place_occluding_vehicles: intersecting, dismiss')
        does_intersect = True
        break
    if not does_intersect:
      vehicles.append(vehicle)
      logging.debug ('place_occluding_vehicles: placed this one')

  return vehicles


def write_visible_mask (patch_dir):
  '''Return the mask of the car visible area.
  Also write the mask for debugging.
  '''
  logging.debug ('making a mask in %s' % patch_dir)
  mask_path = op.join(patch_dir, 'mask.png')

  depth_all = imread(op.join(patch_dir, 'depth-all.png'))
  depth_car = imread(op.join(patch_dir, 'depth-car.png'))
  assert depth_all is not None, op.join(patch_dir, 'depth-all.png')
  assert depth_car is not None, op.join(patch_dir, 'depth-car.png')

  # full main car mask (including occluded parts)
  mask_car = (depth_car < 255*255)

  bbox = mask2bbox(mask_car)
  if bbox is None:
    raise Exception('Mask is empty. Car is outside of the image.')

  # main_car mask of visible regions
  visible_car = depth_car == depth_all
  un_mask_car = np.logical_not(mask_car)
  visible_car[un_mask_car] = False
  imwrite(mask_path, visible_car.astype(np.uint8)*255)
  return visible_car, bbox


def get_road_badness_mask (patch_dir):
  ''' The area under the road plane is a bad area that should not be seen.
  '''
  logging.debug ('making a road badness mask in %s' % patch_dir)
  badness_path = op.join(patch_dir, 'badness-road.png')

  depth_road = imread(op.join(patch_dir, 'depth-road.png'))
  assert depth_road is not None, op.join(patch_dir, 'depth-road.png')
  mask_road = (depth_road < 255*255)

  def write_and_return_all_good_mask():
    badness_mask = np.zeros(mask_road.shape, dtype=np.uint8)
    imwrite(badness_path, badness_mask)
    return badness_mask
  
  # Find the area under the road.
  background_pixels = np.nonzero(mask_road == 0)
  # If road takes all the image.
  if len(background_pixels[0]) == 0:
    return write_and_return_all_good_mask()
  last_background_pixel = background_pixels[0][-1], background_pixels[1][-1]
  # If the last background pixel is above the road, there's no badness.
  if last_background_pixel[0] < mask_road.shape[0] - 1:
    return write_and_return_all_good_mask()
  else:
    # Use cv2.floodFill to fill the whole area around last_background_pixel.
    seed_point = last_background_pixel[1], last_background_pixel[0]
    badness_mask = np.zeros((mask_road.shape[0] + 2, mask_road.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(mask_road.astype(np.uint8), mask=badness_mask, seedPoint=seed_point, newVal=255)
    # Make it binary first to work round cv2 undocumented behavior.
    badness_mask = (badness_mask.astype(np.uint8) > 0) * 255
    badness_mask = badness_mask[1:-1,1:-1]
    imwrite(badness_path, badness_mask)
    return badness_mask > 0


def get_building_badness_mask (patch_dir):
  ''' The area under the road plane is a bad area that should not be seen.
  '''
  logging.debug ('making a building badness mask in %s' % patch_dir)
  badness_path = op.join(patch_dir, 'badness-building.png')

  depth_building = imread(op.join(patch_dir, 'depth-building.png'))
  assert depth_building is not None, op.join(patch_dir, 'depth-building.png')
  mask_building = (depth_building < 255*255)

  # Find the parts of the scene on two sides of the building.  
  _, labels = cv2.connectedComponents(mask_building.astype(np.uint8))
  logging.info('get_building_badness_mask found %d background components' % labels.max())
  if labels.max() <= 1:
    badness_mask = np.zeros(mask_building.shape, dtype=np.uint8)
  else:
    # Make it simple and mark both sides as bad.
    badness_mask = (labels > 0).astype(np.uint8) * 255

  imwrite(badness_path, badness_mask)
  return badness_mask > 0


def get_visible_perc (patch_dir, visible_car):
  '''Some parts of the main car is occluded. 
  Calculate the percentage of the occluded part.
  Args:
    patch_dir:     dir with files depth-all.png, depth-car.png
  Returns:
    visible_perc:  occluded fraction
  '''
  mask_car = imread(op.join(patch_dir, 'depth-car.png')) < 255*255
  assert visible_car is not None
  assert mask_car is not None

  # visible percentage
  nnz_car     = np.count_nonzero(mask_car)
  nnz_visible = np.count_nonzero(visible_car)
  visible_perc = float(nnz_visible) / nnz_car
  logging.debug ('visible perc: %0.2f' % visible_perc)
  return visible_perc


def process_scene_dir(patch_dir):
  try:
    patch = imread(op.join(patch_dir, 'render.png'))[:,:,:3]
    mask, bbox = write_visible_mask (patch_dir)
    visible_perc = get_visible_perc (patch_dir, mask)
    road_badness_mask = get_road_badness_mask(patch_dir)
    building_badness_mask = get_building_badness_mask(patch_dir)
    badness_mask = np.bitwise_or(road_badness_mask, building_badness_mask)

    out_info = json.load(open( op.join(patch_dir, OUT_INFO_NAME) ))
    bbox = mask2bbox(mask)
    if bbox is None:
      logging.warning('Nothing is visible in the patch, mask2bbox returned None.')
      return None
    name = out_info['model_id']  # Write model_id to name.
    yaw = out_info['azimuth']
    pitch = out_info['altitude']
    color = out_info['color']
    # Convention about how to record bad areas.
    mask = mask.astype(np.uint8) * 255
    np.putmask(mask, badness_mask, 128)
    cv2.imwrite(op.join(patch_dir, 'badness.png'), badness_mask.astype(np.uint8) * 255)
    cv2.imwrite(op.join(patch_dir, 'mask-with-badness.png'), mask)
    return (patch, mask, name, bbox, visible_perc, yaw, pitch, color)
  except:
    logging.error('A patch at %s failed to process: %s' %
                  (patch_dir, traceback.format_exc()))
    return None


def run_patches_job (job):
  WORK_DIR = '%s-%d' % (WORK_PATCHES_DIR, os.getpid())
  if op.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)
  os.makedirs(WORK_DIR)
  log_path = '%s.log' % WORK_PATCHES_DIR

  logging.info ('run_patches_job started job %d' % job['job_id'])

  # After getting info delete to avoid pollution.
  main_model  = job['main_model']
  del job['main_model']
  occl_models = job['occl_models']
  del job['occl_models']
  # place the main vehicle
  main_vehicle = main_model
  main_vehicle.update({'x': 0, 'y': 0, 'azimuth': 90})
  # place occluding vehicles
  occl_vehicles = place_occluding_vehicles (main_vehicle, occl_models)
  logging.info ('have total of %d occluding cars' % len(occl_vehicles))
  # Finally, send to job.
  job['vehicles'] = [main_vehicle] + occl_vehicles
  job['logging'] = logging.getLogger().getEffectiveLevel()

  job_path = op.join(WORK_DIR, JOB_INFO_NAME)
  with open(job_path, 'w') as f:
    logging.debug('writing info to job_path %s' % job_path)
    logging.debug('job:\n%s' % pformat(job))
    f.write(json.dumps(job, indent=2))

  with open(log_path, 'a') as f:
    f.write(json.dumps(job, indent=2))

  if os.getenv('BLENDER_PATH') is None:
    raise Exception('BLENDER_PATH is not in environmental variables.')
  try:
    command = [os.getenv('BLENDER_PATH'), '--background', '--python',
               op.join(op.dirname(os.path.realpath(__file__)), 'photoSession.py')]
    if job['logging'] == 10:
      returncode = subprocess.call (command, shell=False) #, stdout=FNULL, stderr=FNULL)
    else:
      returncode = subprocess.call (command, shell=False, stdout=FNULL)#, stderr=FNULL)
    logging.debug ('blender returned code %s' % str(returncode))
    patch_entries = [process_scene_dir(patch_dir) for
            patch_dir in sorted(glob(op.join(WORK_DIR, '??????')))]
  except:
    logging.error('job for %s failed to process: %s' %
                  (job['vehicles'][0]['model_id'], traceback.format_exc()))
    patch_entries = None

  if not job['save_blender']:
      shutil.rmtree(WORK_DIR)
  return patch_entries


def write_results(dataset_writer, patch_entries, use_90turn):
  if patch_entries is None:
    logging.warning('Dropping the whole scene.')
    return

  if any(x is None for x in patch_entries) and use_90turn:
    logging.error('One patch is bad for some reason, dropping the whole scene.')
    return

  for i,patch_entry in enumerate(patch_entries):
    if patch_entry is not None:
      (patch, mask, name, bbox, visible_perc, yaw, pitch, color) = patch_entry
      imagefile = dataset_writer.addImage(image=patch, mask=mask)
      car = {'imagefile': imagefile, 'name': name, 'score': visible_perc,
          'x1': int(bbox[0]), 'y1': int(bbox[1]), 'width': int(bbox[2]), 'height': int(bbox[3]),
          'yaw': yaw, 'pitch': pitch, 'color': color
          }
      carid = dataset_writer.addObject(car)
      if use_90turn:
        if i % 2 == 0:
          match = dataset_writer.addMatch(carid)
        else:
          dataset_writer.addMatch(carid, match)
    

def _fetch_cad_models(cursor, clause):
  cursor.execute('SELECT collection_id,model_id,dims_L,dims_W,dims_H,color FROM cad %s' % clause)
  models = cursor.fetchall()
  shuffle(models)
  models = [{'collection_id': x[0], 
             'model_id': x[1], 
             'dims': {'x': x[2], 'y': x[3], 'z': x[4]},
             'color': x[5]
            } for x in models]
  return models


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--cad_db_path', required=True)
  parser.add_argument('-o', '--out_db_file', default='data/patches/test/scenes.db')
  parser.add_argument('--logging', type=int, default=20, choices=[10,20,30,40])
  parser.add_argument('--num_sessions',    type=int, default=5)
  parser.add_argument('--num_per_session', type=int, default=2)
  parser.add_argument('--num_occluding',   nargs='+', type=int, default=[1,2,3,4,5])
  parser.add_argument('--mode', default='SEQUENTIAL', choices=['SEQUENTIAL', 'PARALLEL'])
  parser.add_argument('--save_blender', action='store_true',
                      help='save .blend render file')
  parser.add_argument('--clause_main', default='WHERE error IS NULL',
                      help='clause to SQL query to define main (center) model')
  parser.add_argument('--clause_occl', default='WHERE error IS NULL',
                      help='clause to SQL query to define models around the main one')
  parser.add_argument('--azimuth_low', type=float, default=0.)
  parser.add_argument('--azimuth_high', type=float, default=360.)
  parser.add_argument('--pitch_low', type=float, default=20.)
  parser.add_argument('--pitch_high', type=float, default=40.)
  parser.add_argument('--use_90turn', action='store_true')
  args = parser.parse_args()

  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')
  progressbar.streams.wrap_stderr()

  # Get main_models and occl_models_pool from cad_db_path.
  if not op.exists(args.cad_db_path):
    raise Exception('Cad db does not exist at "%s"' % args.cad_db_path)
  cad_conn = sqlite3.connect(args.cad_db_path)
  cad_cursor = cad_conn.cursor()
  main_models = _fetch_cad_models(cad_cursor, args.clause_main)
  logging.info('Using total %d models.' % len(main_models))
  occl_models_pool = _fetch_cad_models(cad_cursor, args.clause_occl)
  logging.info('Using total %d occluding models.' % len(occl_models_pool))
  cad_conn.close()

  job = {'num_per_session': args.num_per_session,
          'azimuth_low':  args.azimuth_low,
          'azimuth_high': args.azimuth_high,
          'pitch_low':    args.pitch_low,
          'pitch_high':   args.pitch_high,
          'save_blender': args.save_blender,
          'use_90turn':   args.use_90turn}

  # give parameters to each job
  jobs = [job.copy() for i in range(args.num_sessions)]
  for i,job in enumerate(jobs):
    job['job_id'] = i
    job['main_model'] = main_models[i % len(main_models)]
    num_occluding = choice(args.num_occluding)  # Randomly pick number of occluding cars.
    job['occl_models'] = sample(occl_models_pool, num_occluding)
    logging.debug(job['occl_models'])

  dataset_writer = DatasetWriter(args.out_db_file,
      rootdir=op.dirname(args.out_db_file), overwrite=True, media='video')

  # workhorse
  progressbar = progressbar.ProgressBar(max_value=len(jobs))
  if args.mode == 'SEQUENTIAL':
    for job in progressbar(jobs):
      patch_entries = run_patches_job (job)
      write_results(dataset_writer, patch_entries, args.use_90turn)
  elif args.mode == 'PARALLEL':
    pool = multiprocessing.Pool(processes=5)
    logging.info ('the pool has %d workers' % pool._processes)
    for patch_entries in progressbar(pool.imap(run_patches_job, jobs)):
      write_results(dataset_writer, patch_entries, args.use_90turn)
    pool.close()
    pool.join()

  dataset_writer.close()

