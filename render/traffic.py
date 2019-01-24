import sys, os, os.path as op
sys.path.insert(0, os.path.join(os.getenv('CITY_PATH'), 'src'))
import json
from math import cos, sin, pi, sqrt, pow
import numpy as np
from scipy.misc import imresize
import cv2
import string
import logging
from datetime import datetime
import time, hashlib
import copy
import cPickle
from numpy.random import normal, uniform, choice
from learning.helperSetup import setupLogging, setParamUnlessThere
from learning.dbUtilities import gammaProb
from Cad import Cad
from Camera import Camera
from Video import Video

from renderUtil import atcadillac

WORK_RENDER_DIR     = atcadillac('blender/current-frame')
TRAFFIC_FILENAME  = 'traffic.json'


'''
Distribute cars across the map according to the lanes map and model collections
'''


hash_generator = hashlib.sha1()


def axes_png2blender (points, origin, pxls_in_meter):
    '''Change coordinate frame from pixel-based to blender-based (meters)
    Args:
      origin   - a dict with 'x' and 'y' fields, will be subtracted from each point
      pxls_in_meter - a scalar, must be looked up at the map image
    Returns:
      nothing
    '''
    assert points, 'there are no points'
    assert origin is not None and 'x' in origin and 'y' in origin, origin
    assert pxls_in_meter is not None
    for point in points:
        logging.debug ('axes_png2blender: before x,y = %f,%f' % (point['x'], point['y']))
        point['x'] = (point['x'] - origin['x']) / pxls_in_meter
        point['y'] = -(point['y'] - origin['y']) / pxls_in_meter
        logging.debug ('axes_png2blender: after  x,y = %f,%f' % (point['x'], point['y']))



def m2i (dist_m, N, L_m):
  ''' Unit conversion from meters to map points 
  Args:
    N       - number of points in the map
    L_m     - length of the lane in maters
  '''
  IPM = N / L_m
  return dist_m * IPM


def kph2ipf (speed_kph, N, L_m, avg_fps):
  ''' Unit convertion from kilometers-per-hour to i-per-frame, 
        where 'i' is the point index in the map
  Args:
    N       - number of points in the map
    L_m     - length of the lane in maters
    avg_fps - frames-per-second of generated video (~2 in NYC videos)
  '''
  IPM = N / L_m
  speed_mps = speed_kph / 3.6
  speed_mpf = speed_mps / avg_fps
  speed_ipf = speed_mpf * IPM
  return speed_ipf


def fit_image_to_screen(image, screen=(900,1440), fraction=0.5):
  h, w = image.shape[:2]
  fraction = min(screen[0] / float(h) * fraction, screen[1] / float(w) * fraction)
  return imresize(image, fraction)


class Sun:
  def __init__(self):
    sun_pose_file  = 'resources/sun_position.pkl'

    with open(atcadillac(sun_pose_file), 'rb') as f:
      self.data = cPickle.load(f)

  def __getitem__(self, time):
    time = datetime (year=2015, month=time.month, day=time.day,
                     hour=time.hour, minute=time.minute)

    sun_pose = self.data[time]
    return {'altitude': sun_pose[0], 'azimuth': sun_pose[1]}




class Vehicle:
  ''' Vehicle class stores info in a dict. '''

  def __init__(self, model_info):

    # model
    self.info = model_info
    
    # id
    hash_generator.update(str(time.time()))
    self.info['id'] = hash_generator.hexdigest()[:10]

    # traffic params
    self.info['ipoint'] = 0
    self.info['speed_dev'] = np.random.normal(scale=0.5)

  def __getitem__(self, key):
      return self.info[key]

  def __setitem__(self, key, value):
      self.info[key] = value

  def __contains__(self, key):
      return True if key in self.info else False


class Lane:
  ''' Lane is responsible for moving all the vehicle in this lane. '''

  def __init__(self, name, lane_dict, cad, speed_kph, pxls_in_meter):
    self.name = name

    self.N   = lane_dict['N']
    self.L_m = lane_dict['length'] / pxls_in_meter
    logging.debug('lane %s has %d points, length %.0f m' % (name, self.N, self.L_m))

    x        = lane_dict['x']
    y        = lane_dict['y']
    azimuth  = lane_dict['azimuth']
    self.points = [[x[i], y[i], azimuth[i]] for i in range(self.N)]

    self.step = 0
    self.vehicles = []
    self.cad = cad
    self.intercar_m = 5 + speed_kph / 3.6 * 3 # "count one thousand one, ..."
    self.speed_kph = speed_kph
    self.lane_speed_dev = 0.

    model_info = self.cad.get_random_ready_models(number=1)[0]
    self.vehicles.append(Vehicle(model_info))


  def update(self):
    '''Move all the cars, maybe start a new car, remove one if it jumps off
    The model for moving along the lane is:
      traffic_speed = v + sin(traffic_v_omega time)
      car_position += traffic_speed + sin(car_v_omega time) * intercar_dist/3
    '''
    lane_speed_ipf = kph2ipf (self.speed_kph, self.N, self.L_m, avg_fps=2.)
    self.lane_speed_dev += np.mean(np.random.poisson(0.5, 10)) - 0.5
    self.lane_speed_dev *= 0.5
    lane_speed_ipf *= (1 + 1. * self.lane_speed_dev)
    logging.debug('lane %s, speed_dev %0.2f' % (self.name, self.lane_speed_dev))

    # do we need to add another
    dist_behind_m = self.vehicles[0]['dims']['x'] + self.intercar_m
    dist_behind_points = m2i(dist_behind_m, self.N, self.L_m)
    if len(self.vehicles) == 0 or self.vehicles[0]['ipoint'] > dist_behind_points:
      self.vehicles.insert (0, Vehicle(self.cad))
    
    # update all cars
    for vehicle in self.vehicles:
      vehicle['speed_dev'] += np.mean(np.random.poisson(0.5, 10)) - 0.5
      vehicle['speed_dev'] *= 0.5
      vehicle_speed_ipf = lane_speed_ipf * (1 + 1. * vehicle['speed_dev'])
      vehicle['ipoint'] += int(vehicle_speed_ipf)
      
      #logging.debug('updated vehicle: %s. Its n is %0.2f' % 
      #               (vehicle['id'], vehicle['ipoint']))

      # mark the vehicle for deletion if necessary
      if vehicle['ipoint'] >= self.N:
        vehicle['to_delete'] = True
      else:
        point = self.points[vehicle['ipoint']]
        vehicle['x'] = point[0]
        vehicle['y'] = point[1]
        vehicle['azimuth'] = point[2]

    # delete those vehicles that are marked
    self.vehicles = [v for v in self.vehicles if not 'to_delete' in v]

    self.step += 1


class TrafficModel:
  ''' Traffic consists of several lanes.
  Each lane is completely independent, vehicles do not change lanes. 
  '''

  def __init__(self, camera, video, cad, speed_kph, burn_in=True):

    self.sun = Sun()  # TODO: only load when it's sunny

    self.camera = camera
    self.video = video

    # load mask
    if 'mask' in camera:
      mask_path = atcadillac(op.join(camera['camera_dir'], camera['mask']))
      self.mask = cv2.imread (mask_path, cv2.IMREAD_GRAYSCALE)
      assert self.mask is not None, mask_path
      logging.info ('TrafficModel: loaded a mask')
    else:
      self.mask = None

    # create lanes
    lanes_path = atcadillac(op.join(camera['camera_dir'], camera['lanes_name']))
    lanes_dicts = json.load(open( lanes_path ))
    self.lanes = [Lane(('%d' % i), l, cad, speed_kph, camera['pxls_in_meter']) 
                  for i,l in enumerate(lanes_dicts)]
    logging.info ('TrafficModel: loaded %d lanes' % len(self.lanes))

    if burn_in:
      for i in range(100): self.get_next_frame(time=datetime.now())


  def _collect_vehicles (self):
    ''' collect vehicles from each lane and apply the mask '''

    # collect cars from all lanes
    vehicles = []
    for lane in self.lanes:
      vehicles += lane.vehicles

    # if mask exists, leave only cars in the mask
    if self.mask is not None:
      vehicles = [v for v in vehicles if self.mask[v['y'],v['x']]]

    return vehicles


  def generate_map (self):
    ''' generate lanes map with cars as icons for visualization '''

    width  = self.camera['map_dims']['width']
    height = self.camera['map_dims']['height']

    # generate maps
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for lane in self.lanes:
      for i,point in enumerate(lane.points):
        img[point[1], point[0], 1] = 255
        # direction
        img[point[1], point[0], 2] = i * 255 / lane.N
        # azimuth
        img[point[1], point[0], 0] = point[2]
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    # put cars on top
    for v in self._collect_vehicles():
      cv2.circle(img, (v['x'],v['y']), 5, (128,128,128), -1)

    return img


  def get_next_frame(self, time):

    for lane in self.lanes:
      lane.update()

    vehicles_blender = [{x: v[x] for x in 
          ['x', 'y', 'azimuth', 'model_id', 'collection_id', 'vehicle_type']}
          for v in self._collect_vehicles()]

    axes_png2blender (vehicles_blender, 
                      self.camera['origin_image'], self.camera['pxls_in_meter'])

    # figure out sun position based on the timestamp
    logging.info ('received timestamp: %s' % time)
    sun_pose = self.sun[time]
    logging.info ('calculated sunpose: %s' % str(sun_pose))

    traffic = {'sun_altitude': sun_pose['altitude'],
               'sun_azimuth':  sun_pose['azimuth'],
               'weather':      self.video['weather'],
               'vehicles':     vehicles_blender
               }

    return traffic
    



def sq(x): return pow(x,2)

def get_norm(x): return sqrt (sq(x['x']) + sq(x['y']) + sq(x['z']))


class TrafficModelRandom:

  def __init__ (self, camera, video, cad, num_cars_mean):

    self.sun = Sun()

    self.camera = camera
    self.video = video

    # get the map of azimuths. 
    # it has gray values (r==g==b=) and alpha, saved as 4-channels
    azimuth_path = atcadillac(op.join(camera['camera_dir'], camera['azimuth_name']))
    azimuth_map = cv2.imread (azimuth_path, cv2.IMREAD_UNCHANGED)
    assert azimuth_map is not None and azimuth_map.shape[2] == 4

    # black out the invisible azimuth_map regions
    if 'mask' in camera and camera['mask']:
      mask_path = atcadillac(op.join(camera['camera_dir'], camera['mask']))
      mask = cv2.imread (mask_path, cv2.IMREAD_GRAYSCALE)
      assert mask is not None, mask_path
      azimuth_map[mask] = 0

    self.num_cars_mean = num_cars_mean
    self.num_cars_std  = num_cars_mean * 1.0
    self.azimuth_map = azimuth_map
    self.cad = cad
    self.pxls_in_meter = camera['pxls_in_meter']


  def _put_random_vehicles (self):
    '''Places a number of random models to random points in the lane map.

    azimuth_map:         a color array (all values are gray) with alpha mask, [YxXx4]
    pxl_in_meter:        for this particular map
    num_cars_mean:       the average number of vehicles to pick
    num_cars_std:        std of the average number of cars to pick
    intercar_dist_mult:  cars won't be sampled closer than sum of their dims, 
                         multiplied by this factor
    Returns:
      vehicles:          a list of dictionaries, each has x,y,azimuth attributes
    '''
    INTERCAR_DIST_MULT = 1.5

    # make azimuth_map a 2D array
    azimuth_map = self.azimuth_map.copy()
    alpha, azimuth_map = azimuth_map[:,:,-1], azimuth_map[:,:,0]

    # get indices of all points which are non-zero
    Ps = np.transpose(np.nonzero(alpha))
    assert Ps.shape[0] > 0, 'azimuth_map is all zeros'

    # pick random points
    #num_cars = gammaProb (, self.num_cars_mean, 1)[0]
    num_cars = int(np.random.normal(loc=self.num_cars_mean, scale=self.num_cars_std))
    print num_cars
    if num_cars <= 0: num_cars = 1
    ind = np.random.choice (Ps.shape[0], size=num_cars, replace=True)

    # get angles (each azimuth is multiplied by 2 by convention)
    dims_dict = {}
    vehicles = []
    for P in Ps[ind]:
        x = P[1]
        y = P[0]
        azimuth = azimuth_map[y][x] * 2
        logging.debug ('put_random_vehicles x: %f, y: %f, azimuth: %f' % (x, y, azimuth))

        # car does not need to be in the lane center
        pos_std = 0.2   # meters away from the middle of the lane
        x += np.random.normal(0, self.pxls_in_meter * pos_std)
        y += np.random.normal(0, self.pxls_in_meter * pos_std)

        x = int(x)
        y = int(y)

        # keep choosing a car until find a valid one
        vehicle = self.cad.get_random_ready_models(number=1)[0]
        dims_dict[vehicle['model_id']] = vehicle['dims']

        # cars can't be too close. TODO: they can be close on different lanes
        too_close = False
        for vehicle2 in vehicles:

            # get the minimum idstance between cars in pixels
            car1_sz = get_norm(dims_dict[vehicle['model_id']])
            car2_sz = get_norm(dims_dict[vehicle2['model_id']])
            min_intercar_dist_pxl = INTERCAR_DIST_MULT * self.pxls_in_meter * (car1_sz + car2_sz) / 2

            if sqrt(sq(vehicle2['y']-y) + sq(vehicle2['x']-x)) < min_intercar_dist_pxl:
                too_close = True
        if too_close: 
            continue
        
        vehicle['x'] = x
        vehicle['y'] = y
        vehicle['azimuth'] = azimuth
        vehicles.append(vehicle)

    print 'wrote %d vehicles' % len(vehicles)
    return vehicles


  def get_next_frame (self, time):

    self.vehicles = self._put_random_vehicles()
    vehicles_blender = [{x: v[x] for x in 
          ['x', 'y', 'azimuth', 'model_id', 'collection_id', 'vehicle_type']}
          for v in self.vehicles]


    axes_png2blender (vehicles_blender, 
                      self.camera['origin_image'], self.camera['pxls_in_meter'])

    # figure out sun position based on the timestamp
    logging.info ('received timestamp: %s' % time)
    sun_pose = self.sun[time]
    logging.info ('calculated sunpose: %s' % str(sun_pose))

    traffic = {'sun_altitude': sun_pose['altitude'],
               'sun_azimuth':  sun_pose['azimuth'],
               'weather':      self.video['weather'],
               'vehicles':     vehicles_blender
               }

    return traffic


  def generate_map (self):
    ''' generate lanes map with cars as icons for visualization '''

    # make azimuth_map a 2D array
    img = self.azimuth_map.copy()
    #alpha, azimuth_map = azimuth_map[:,:,-1], azimuth_map[:,:,0]

    # put cars on top
    for v in self.vehicles:
      cv2.circle(img, (v['x'],v['y']), 5, (128,128,128), -1)

    return img





if __name__ == "__main__":

  setupLogging ('log/augmentation/traffic.log', logging.DEBUG, 'w')

  video_dir = 'augmentation/scenes/cam166/Feb23-09h'
  collection_names = ['7c7c2b02ad5108fe5f9082491d52810', 
                      'uecadcbca-a400-428d-9240-a331ac5014f6']
  timestamp = datetime.now()
  video = Video(video_dir)
  camera = video.build_camera()

  cad = Cad(collection_names)

  #model = TrafficModel (camera, video, cad=cad, speed_kph=10, burn_in=True)
  model = TrafficModelRandom (camera, video, cad, num_cars_mean=10)

  # cv2.imshow('lanesmap', model.generate_map())
  # cv2.waitKey(-1)
  while True:
    model.get_next_frame(timestamp)
    display = fit_image_to_screen(model.generate_map())
    cv2.imshow('lanesmap', display)
    key = cv2.waitKey(-1)
    if key == 27: break

  # traffic_path = op.join(WORK_RENDER_DIR, TRAFFIC_FILENAME)
  # with open(traffic_path, 'w') as f:
  #     f.write(json.dumps(traffic, indent=4))
