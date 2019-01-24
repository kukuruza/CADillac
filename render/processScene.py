import sys, os, os.path as op
sys.path.insert(0, op.join(os.getenv('CITY_PATH'), 'src'))
from glob import glob
from time import sleep, time
import scipy.misc
import json
import numpy as np
import cv2
import argparse
import logging
import sqlite3
import subprocess
import multiprocessing
import shutil
from datetime import datetime, timedelta
from math import pi, atan, atan2, pow, sqrt, ceil

from learning.dbUtilities import *
from learning.helperImg import SimpleWriter, ReaderVideo
from learning.helperSetup import _setupCopyDb_, setupLogging, dbInit
from learning.helperSetup import setParamUnlessThere, assertParamIsThere
from monitor.MonitorDatasetClient import MonitorDatasetClient
from Cad import Cad
from Camera import Camera
from Video import Video
from traffic import TrafficModel
from colorCorrection import color_correction, unsharp_mask

from renderUtil import atcadillac



WORK_RENDER_DIR     = atcadillac('blender/current-frame')
RENDERED_FILENAME   = 'render.png'
CARSONLY_FILENAME   = 'cars-only.png'
BACKGROUND_FILENAME = 'background.png'
COMBINED_FILENAME   = 'out.png'
MASK_FILENAME       = 'mask.png'
TRAFFIC_FILENAME    = 'traffic.json'
CORRECTION_FILENAME = 'color-correction.json'
FNULL = open(op.join(os.getenv('CITY_PATH'), 'log/augmentation/blender.log'), 'w')

assert os.getenv('BLENDER_ROOT') is not None, \
    'export BLENDER_ROOT with path to blender binary as environmental variable'


def _sq(x): return pow(x,2)

def _get_norm_xy_(a): return sqrt(_sq(a['x']) + _sq(a['y']))


def extract_bbox (depth):
  '''Extract a single (if any) bounding box from the image
  Args:
    depth: has only one (or no) car in the image.
  Returns:
    bbox:  (x1, y1, width, height)
  '''
  # keep only vehicles with resonable bboxes
  if np.count_nonzero(depth < 255) == 0:   # or are there any artifacts
    return None

  # get bbox
  nnz_indices = np.argwhere(depth < 255)
  (y1, x1), (y2, x2) = nnz_indices.min(0), nnz_indices.max(0) + 1 
  (height, width) = y2 - y1, x2 - x1
  return (x1, y1, width, height)


def extract_annotations (work_dir, c, cad, camera, imagefile, monitor=None):
  '''Parse output of render and all metadata into our SQL format.
  This function knows about SQL format.
  Args:
      work_dir:         path with depth-s
      c:                cursor to existing db in our format
      cad:              info on the pose of every car in the frame, 
                        and its id within car collections
      camera:           dict of camera height and orientation
      imagefile:        database entry
      monitor:          MonitorDatasetClient object for uploading vehicle info
  Returns:
      nothing
  '''
  traffic = json.load(open( op.join(work_dir, TRAFFIC_FILENAME) ))

  for i,vehicle in enumerate(traffic['vehicles']):

    # get bbox
    depth_path = op.join (work_dir, 'depth-%03d.png' % i)
    assert op.exists(depth_path), depth_path
    depth = cv2.imread(depth_path, 0)
    bbox = extract_bbox (depth)
    if bbox is None: continue

    # get vehicle "name" (that is, type)
    model = cad.get_model_by_id_and_collection (vehicle['model_id'], 
                                                vehicle['collection_id'])
    assert model is not None
    name = model['vehicle_type']

    # get vehicle angles (camera roll is assumed small and ignored)
    azimuth_view = -atan2(vehicle['y'], vehicle['x']) * 180 / pi
    yaw = (180 - vehicle['azimuth'] + azimuth_view) % 360
    height = camera.info['origin_blender']['z']
    pitch = atan(height / _get_norm_xy_(vehicle)) * 180 / pi

    # get vehicle visibility
    vis = vehicle['visibility']

    # put all info together and insert into the db
    entry = (imagefile, name, bbox[0], bbox[1], bbox[2], bbox[3], yaw, pitch, vis)
    c.execute('''INSERT INTO cars(imagefile,name,x1,y1,width,height,yaw,pitch,score)
                 VALUES (?,?,?,?,?,?,?,?,?);''', entry)

    if monitor is not None:
      monitor.upload_vehicle({'vehicle_type': name, 'yaw': yaw, 'pitch': pitch,
                              'width': bbox[2], 'height': bbox[3]})



def _get_visible_perc (patch_dir):
  '''Some parts of the main car is occluded. 
  Calculate the percentage of the occluded part.
  Args:
    patch_dir:     dir with files depth-all.png, depth-car.png
  Returns:
    visible_perc:  occluded fraction
  '''
  visible_car = cv2.imread(op.join(patch_dir, 'mask.png'), 0) > 0
  mask_car    = cv2.imread(op.join(patch_dir, 'depth-car.png'), -1) < 255*255
  assert visible_car is not None
  assert mask_car is not None

  # visible percentage
  nnz_car     = np.count_nonzero(mask_car)
  nnz_visible = np.count_nonzero(visible_car)
  visible_perc = float(nnz_visible) / nnz_car
  logging.debug ('visible perc: %0.2f' % visible_perc)
  return visible_perc




def _get_masks (patch_dir, frame_info):
  ''' patch_dir contains "depth-all.png" and a bunch of "depth-XXX.png"
  Compare them and make a final mask of each car.
  This function changes 'frame_info'
  '''

  # read depth-all
  depth_all_path = op.join(patch_dir, 'depth-all.png')
  depth_all = cv2.imread(depth_all_path, 0)
  assert depth_all is not None, depth_all_path
  assert depth_all.dtype == np.uint8

  # mask for all cars
  mask_all = np.zeros(depth_all.shape, dtype=np.uint8)

  for i in range(len(frame_info['vehicles'])):
    # read depth-XXX and check
    depth_car_path = op.join(patch_dir, 'depth-%03d.png' % i)
    depth_car = cv2.imread(depth_car_path, 0)
    assert depth_car is not None, depth_car_path
    assert depth_car.dtype == np.uint8
    assert depth_car.shape == depth_all.shape

    # get mask of the car
    mask_full = depth_car < 255
    mask_visible = np.bitwise_and (mask_full, depth_car == depth_all)
    color = 255 * i / len(frame_info['vehicles'])
    mask_all += mask_visible.astype(np.uint8) * color
    #cv2.imshow('mask_full', mask_full.astype(np.uint8) * 255)
    #cv2.imshow('mask_visible', mask_visible.astype(np.uint8) * 255)
    #cv2.imshow('mask_all', mask_all)
    #cv2.waitKey(-1)

    # find the visibility percentage of the car
    if np.count_nonzero(mask_full) == 0:
        visibility = 0
    else:
        visibility = float(np.count_nonzero(mask_visible)) \
                   / float(np.count_nonzero(mask_full))
    frame_info['vehicles'][i]['visibility'] = visibility

  # disabled a mask output segmented by car. Returning a binary mask now.
  #return mask_all
  return mask_all > 0



def render_frame (video, camera, traffic):
  ''' Write down traffci file for blender and run blender with renderScene.py 
  All work is in current-frame dir.
  '''
  WORK_DIR = '%s-%d' % (WORK_RENDER_DIR, os.getpid())
  setParamUnlessThere (traffic, 'save_blender_files', False)
  setParamUnlessThere (traffic, 'render_individual_cars', True)
  unsharp_mask_params = {'radius': 4.7, 'threshold': 23, 'amount': 1}

  # load camera dimensions (compare it to everything for extra safety)
  width0  = camera.info['camera_dims']['width']
  height0 = camera.info['camera_dims']['height']
  logging.debug ('camera width,height: %d,%d' % (width0, height0))

  image = None
  mask = None

  # pass traffic info to blender
  traffic['scale'] = camera.info['scale']
  traffic_path = op.join(WORK_DIR, TRAFFIC_FILENAME)
  if not op.exists(op.dirname(traffic_path)):
    os.makedirs(op.dirname(traffic_path))
  with open(traffic_path, 'w') as f:
    f.write(json.dumps(traffic, indent=4))

  # remove so that they do not exist if blender fails
  if op.exists(op.join(WORK_DIR, RENDERED_FILENAME)):
      os.remove(op.join(WORK_DIR, RENDERED_FILENAME))
  if op.exists(op.join(WORK_DIR, 'depth-all.png')):
      os.remove(op.join(WORK_DIR, 'depth-all.png'))
  # render
  assert video.render_blend_file is not None
  render_blend_path = atcadillac(video.render_blend_file)
  command = ['%s/blender' % os.getenv('BLENDER_ROOT'), render_blend_path, 
             '--background', '--python',
             '%s/src/augmentation/renderScene.py' % os.getenv('CITY_PATH')]
  logging.debug ('WORK_DIR: %s' % WORK_DIR)
  logging.debug (' '.join(command))
  returncode = subprocess.call (command, shell=False, stdout=FNULL, stderr=FNULL)
  logging.info ('rendering: blender returned code %s' % str(returncode))

  # check and sharpen rendered
  rendered_filepath = op.join(WORK_DIR, RENDERED_FILENAME)
  image = cv2.imread(rendered_filepath, -1)
  assert image is not None
  assert image.shape == (height0, width0, 4), image.shape
  image = unsharp_mask (image, unsharp_mask_params)
  cv2.imwrite (rendered_filepath, image)

  # check and sharpen cars-only
  carsonly_filepath = op.join(WORK_DIR, CARSONLY_FILENAME)
  image = cv2.imread(carsonly_filepath, -1)
  assert image is not None
  assert image.shape == (height0, width0, 4), image.shape
  image = unsharp_mask (image, unsharp_mask_params)
  shutil.move (carsonly_filepath, op.join(WORK_DIR, 'unsharpened.png'))
  cv2.imwrite (carsonly_filepath, image)

  # create mask
  if traffic['render_individual_cars'] == True:
    mask = _get_masks (WORK_DIR, traffic)
    # TODO: visibility is returned via traffic file, NOT straightforward
    with open(traffic_path, 'w') as f:
        f.write(json.dumps(traffic, indent=4))

  # correction_path = op.join(WORK_DIR, CORRECTION_FILENAME)
  # if op.exists(correction_path): os.remove(correction_path)
  # if not params['no_correction']:
  #     correction_info = color_correction (video.example_background, background)
  #     with open(correction_path, 'w') as f:
  #         f.write(json.dumps(correction_info, indent=4))

  return image, mask


def combine_frame (background, video, camera):
  ''' Overlay image onto background '''
  jpg_qual = 40

  WORK_DIR = '%s-%d' % (WORK_RENDER_DIR, os.getpid())

  # load camera dimensions (compare it to everything for extra safety)
  width0  = camera.info['camera_dims']['width']
  height0 = camera.info['camera_dims']['height']

  # get background file
  assert background is not None
  assert background.shape == (height0, width0, 3), background.shape
  # make a completely gray background frame hahaha
  #background.fill(128)
  cv2.imwrite (op.join(WORK_DIR, BACKGROUND_FILENAME), background)

  # get shadows file
  #shadow_path = op.join(WORK_DIR, 'render.png')
  #shadow = scipy.misc.imread(shadow_path)
  #shadow[:,:,3] = 0  # assign full transparency 
  #scipy.misc.imsave(shadow_path, shadow)

  # remove previous result so that there is an error if blender fails
  if op.exists(op.join(WORK_DIR, COMBINED_FILENAME)): 
      os.remove(op.join(WORK_DIR, COMBINED_FILENAME))

  # overlay
  assert video.combine_blend_file is not None
  combine_scene_path = atcadillac(video.combine_blend_file)
  command = ['%s/blender' % os.getenv('BLENDER_ROOT'), combine_scene_path,
             '--background', '--python',
             '%s/src/augmentation/combineScene.py' % os.getenv('CITY_PATH')]
  returncode = subprocess.call (command, shell=False, stdout=FNULL, stderr=FNULL)
  logging.info ('combine: blender returned code %s' % str(returncode))
  combined_filepath = op.join(WORK_DIR, COMBINED_FILENAME)
  assert op.exists(combined_filepath), combined_filepath
  image = cv2.imread(combined_filepath)
  assert image.shape == (height0,width0,3), '%s vs %s' % (image.shape, (height0,width0))

  # reencode to match jpeg quality
  shutil.move (combined_filepath, op.join(WORK_DIR, 'uncompressed.png'))
  _, ajpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpg_qual])
  image = cv2.imdecode(ajpg, cv2.CV_LOAD_IMAGE_COLOR)
  cv2.imwrite (combined_filepath, image)

  return image


class Diapason:

  def _parse_range_str_ (self, range_str, length):
    '''Parses string into python range.
    Has another forth number (default to 1), signifying the number of repetitions.
    '''
    assert isinstance(range_str, basestring)
    # remove [ ] around the range
    if len(range_str) >= 2 and range_str[0] == '[' and range_str[-1] == ']':
        range_str = range_str[1:-1]
    # split into three elements start,end,step. Assign step=1 if missing
    arr = range_str.split(':')
    assert len(arr) >= 2 and len(arr) <= 4, 'need 1, 2 or 3 columns ":" in range str'
    if len(arr) == 2: arr.append('1')
    if len(arr) == 3: arr.append('1')
    if arr[0] == '': arr[0] = '0'
    if arr[1] == '': arr[1] = str(length)
    if arr[2] == '': arr[2] = '1'
    if arr[3] == '': arr[3] = '1'
    start = int(arr[0])
    end   = int(arr[1])
    step  = int(arr[2])
    repeatitions = int(arr[3])
    range_py = range(start, end, step)
    range_py = range_py * repeatitions
    logging.info ('Diapason parsed range_str %s into range of length %d' % 
                    (range_str, len(range_py)))
    return range_py

  def __init__ (self, length, frame_range_str):
    self.frame_range = self._parse_range_str_ (frame_range_str, length)

  def intersect (self, diapason):
    interset = set(self.frame_range).intersection(diapason.frame_range)
    self.frame_range = sorted(interset)
    logging.info ('Diapason intersection has %d frames' % len(self.frame_range))
    logging.debug ('Diapason intersection is range %s' % self.frame_range)
    return self

  def frame_range_as_chunks (self, chunk_size):
    ''' Cutting frame_range into chunks for parallel execution '''
    chunks = []
    chunk_num = int(ceil( len(self.frame_range) / float(chunk_size) ))
    for i in range(chunk_num):
      r = self.frame_range[i*chunk_size : min((i+1)*chunk_size, len(self.frame_range))]
      chunks.append(r)
    return chunks



def worker((video, camera, traffic, back, job)):
  ''' wrapper for parallel processing. Argument is an element of frame_jobs 
  '''
  WORK_DIR = '%s-%d' % (WORK_RENDER_DIR, os.getpid())
  if not op.exists(WORK_DIR): os.makedirs(WORK_DIR)

  _, out_mask = render_frame(video, camera, traffic)
  out_image = combine_frame(back, video, camera)

  return out_image, out_mask, WORK_DIR



def sequentialworker(frame_jobs):
  ''' Wrap mywrapper for sequential run (in debugging) '''
  for frame_job in frame_jobs:
    yield worker(frame_job)



def process_video (job):

  assertParamIsThere  (job, 'video_dir')
  video = Video(video_dir=job['video_dir'])
  camera = video.build_camera()

  # some parameters
  assertParamIsThere  (job, 'traffic_file')
  setParamUnlessThere (job, 'save_blender_files', False)
  setParamUnlessThere (job, 'out_video_dir', 
      op.join('augmentation/video', 'cam%s' % camera.info['cam_id'], video.info['video_name']))
  setParamUnlessThere (job, 'no_annotations', False)
  setParamUnlessThere (job, 'timeout', 1000000000)
  setParamUnlessThere (job, 'frame_range', '[::]')
  setParamUnlessThere (job, 'save_blender_files', False)
  job['render_individual_cars'] = not job['no_annotations']

  # load camera dimensions (compare it to everything for extra safety)
  width0  = camera.info['camera_dims']['width']
  height0 = camera.info['camera_dims']['height']

  # for checking timeout
  start_time = datetime.now()

  cad = Cad()

  # upload info on parsed vehicles to the monitor server
  monitor = None # MonitorDatasetClient (cam_id=camera.info['cam_id'])

  # load traffic info
  traffic_video = json.load(open(atcadillac(job['traffic_file'])))
  
  # reader and writer
  video_reader = ReaderVideo()
  image_vfile = op.join(job['out_video_dir'], 'image.avi')
  mask_vfile  = op.join(job['out_video_dir'], 'mask.avi')
  video_writer = SimpleWriter(image_vfile, mask_vfile, {'unsafe': True})

  (conn, c) = dbInit (traffic_video['in_db_file'], op.join(job['out_video_dir'], 'traffic.db'))
  c.execute('SELECT imagefile,maskfile,width,height,time FROM images')
  image_entries = c.fetchall()
  c.execute('DELETE FROM images')

  #assert len(traffic_video['frames']) >= len(image_entries), \
  #  'traffic json is too small %d < %d' % (len(traffic_video['frames']), len(image_entries))

  diapason = Diapason(len(image_entries), job['frame_range'])
  
  num_processes = int(multiprocessing.cpu_count() / 2 + 1)
  pool = multiprocessing.Pool (processes=num_processes)

  # each frame_range chunk is processed in parallel
  for frame_range in diapason.frame_range_as_chunks(pool._processes):
    logging.info ('chunk of frames %d to %d' % (frame_range[0], frame_range[-1]))

    # quit, if reached the timeout
    time_passed = datetime.now() - start_time
    logging.info ('passed: %s' % time_passed)
    if (time_passed.total_seconds() > job['timeout'] * 60):
      logging.warning('reached timeout %d. Passed %s' % (job['timeout'], time_passed))
      break

    # collect frame jobs
    frame_jobs = []
    for frame_id in frame_range:

      (in_backfile, in_maskfile, width, height, _) = image_entries[frame_id]
      assert (width0 == width and height0 == height), (width0, width, height0, height)
      logging.info ('collect job for frame number %d' % frame_id)

      back = video_reader.imread(in_backfile)

      traffic = traffic_video['frames'][frame_id]
      assert traffic['frame_id'] == frame_id, '%d vs %d' % (traffic['frame_id'], frame_id)
      traffic['save_blender_files'] = job['save_blender_files']

      frame_jobs.append((video, camera, traffic, back, job))

    #for i, (out_image, out_mask, work_dir) in enumerate(sequentialworker(frame_jobs)):
    for i, (out_image, out_mask, work_dir) in enumerate(pool.imap(worker, frame_jobs)):
      frame_id = frame_range[i]
      logging.info ('processed frame number %d' % frame_id)

      assert out_image is not None and out_mask is not None
      out_imagefile = video_writer.imwrite (out_image)
      out_maskfile  = video_writer.maskwrite (out_mask)
      logging.info('out_imagefile: %s, out_maskfile: %s' % (out_imagefile, out_maskfile))

      # update out database
      (_, _, width, height, time) = image_entries[frame_id]
      c.execute ('INSERT INTO images(imagefile,maskfile,width,height,time) VALUES (?,?,?,?,?)',
                 (out_imagefile, out_maskfile, width, height, time))
      logging.info('wrote frame %d' % c.lastrowid)

      if not job['no_annotations']:
        extract_annotations (work_dir, c, cad, camera, out_imagefile, monitor)

      if not job['save_blender_files']: 
        shutil.rmtree(work_dir)

      conn.commit()
  conn.close()

  pool.close()
  pool.join()
