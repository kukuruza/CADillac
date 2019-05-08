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


import os, os.path as op
import simplejson as json
import argparse
import logging
import numpy as np
import cv2
import sqlite3
from pprint import pprint, pformat
import matplotlib.pyplot as plt
import subprocess
import traceback
from random import shuffle
from scipy.cluster.hierarchy import fcluster, linkage

from collectionUtilities import atcadillac, safeConnect, getExamplePath, getBlendPath, COLLECTION_WORK_DIR
from collectionDb import maybeCreateTableCad, maybeCreateTableClas, getAllCadColumns
if not op.exists(COLLECTION_WORK_DIR):
    os.makedirs(COLLECTION_WORK_DIR)


def _debug_execute(s, args):
  logging.debug('Going to execute "%s" with arguments %s' % (s, str(args)))


def _queryCollectionsAndModels(cursor, clause):
  s = 'SELECT cad.collection_id, cad.model_id FROM cad %s;' % clause
  logging.debug('Will execute: %s' % s)
  cursor.execute(s)
  entries = cursor.fetchall()
  logging.info('Found %d entries' % len(entries))
  return entries


def _renderExample (collection_id, model_id, overwrite):

  example_path = getExamplePath(collection_id, model_id)
  if not overwrite and op.exists(example_path):
    logging.info ('skipping existing example %s' % example_path)
    return

  blend_path = getBlendPath(collection_id, model_id)
  if not op.exists(blend_path):
    logging.info ('skipping non-existing %s' % blend_path)
    return

  model = {
      'model_id': model_id,
      'collection_id': collection_id,
      'blend_file': blend_path,
      'example_file': example_path
  }
  model_path = op.join(COLLECTION_WORK_DIR, 'model.json')
  with open(model_path, 'w') as f:
    f.write(json.dumps(model, indent=2))

  try:
    command = ['%s/blender' % os.getenv('BLENDER_ROOT'), '--background', '--python',
               atcadillac('src/augmentation/collections/renderExample.py')]
    returncode = subprocess.call (command, shell=False)
    logging.info ('blender returned code %s' % str(returncode))
  except:
    logging.error('failed: %s' % traceback.format_exc())
    return


def _importCollection(cursor, collection, overwrite):
  cols = getAllCadColumns()

  for model in collection['vehicles']:
    try:
      model['collection_id'] = collection['collection_id']
      model['dims_L'] = None
      model['dims_W'] = None
      model['dims_H'] = None
      try:
        model['dims_L'] = model['dims_true']['x']
      except:
        pass
      try:
        model['dims_W'] = model['dims_true']['y']
      except:
        pass
      try:
        model['dims_H'] = model['dims_true']['z']
      except:
        pass

      # Convenience variable.
      modelcol = model['model_id'], model['collection_id']

      # Check if the model is already in this collection.
      cursor.execute('SELECT COUNT(1) FROM cad WHERE model_id=? AND collection_id=?', modelcol)
      num_present = cursor.fetchone()[0]
      assert num_present <= 1, 'Primary key restriction: %s' % model['model_id']

      # Check if there is already such a model in a different collection.
      cursor.execute('SELECT collection_id FROM cad WHERE model_id=? AND collection_id!=?', modelcol)
      valid_collection_ids = cursor.fetchall()
      if len(valid_collection_ids) >= 1:
        model['error'] = 'already in collection(s) %s' % str(valid_collection_ids)
        logging.warning('Model %s is %s' % (model['model_id'], model['error']))

      if num_present == 1 and not overwrite:
        logging.warning('Skipping model %s in collection %s, because it is already there.' % modelcol)
        continue

      elif num_present == 0:
        logging.info('Inserting model %s from collection %s' % modelcol)
        s = 'INSERT INTO cad(%s) VALUES (%s)' % (','.join(cols), ','.join(['?'] * len(cols)))

      elif num_present == 1 and overwrite:
        logging.info('Updating model %s from collection %s' % modelcol)
        s = 'UPDATE cad SET %s WHERE model_id=? AND collection_id=?' % ','.join(['%s=?' % c for c in cols[2:]])
        cols = cols[2:] + cols[:2]  # model_id and collection_id are looped to the back of str.

      logging.debug('Will execute: %s' % s)
      # Form an entry of values, valid for both INSERT and UPDATE.
      entry = tuple([model[name] if name in model else None for name in cols])
      logging.debug(str(entry))
      cursor.execute(s, entry)
      
    # To debug a problem in json, we need to print all info before exiting
    except Exception:
      print('Error occured in model: \n%s' % pformat(model))
      traceback.print_exc()
      sys.exit()


def _getDimsFromBlender(cursor, collection_id, model_id):
  ''' For a single model. '''

  if not op.exists(getBlendPath(collection_id, model_id)):
    logging.error('Blend path does not exist.')
    return None

  model = {'blend_file': getBlendPath(collection_id, model_id)}
  model_path = op.join(COLLECTION_WORK_DIR, 'model.json')
  with open(model_path, 'w') as f:
    f.write(json.dumps(model, indent=4))

  try:
    command = ['%s/blender' % os.getenv('BLENDER_ROOT'), '--background', '--python',
               atcadillac('src/augmentation/collections/getDims.py')]
    returncode = subprocess.call (command, shell=False)
    logging.debug('Blender returned code %s' % str(returncode))

    info = json.load(open(model_path))
    dims, x_wheels = info['dims'], info['x_wheels']
    dims_L, dims_W, dims_H = dims['x'], dims['y'], dims['z']

    # Find wheelbase.
    x_wheels = np.array(x_wheels)
    Z = linkage(np.reshape(x_wheels, (len(x_wheels), 1)), 'ward')
    indices = fcluster(Z, 2, criterion='maxclust')
    x_wheel1 = x_wheels[indices == 1].mean()
    x_wheel2 = x_wheels[indices == 2].mean()
    logging.info('Wheel1: %.2f, wheel2: %.2f' % (x_wheel1, x_wheel2))
    wheelbase = np.abs(x_wheel2 - x_wheel1)

    logging.info('collection_id: %s, model_id: %s, L: %.2f, W: %.2f, H: %.2f, wheelbase: %.2f' % 
        (collection_id, model_id, dims_L, dims_W, dims_H, wheelbase))
    return dims_L, dims_W, dims_H, wheelbase

  except:
    logging.error('Failed: %s' % traceback.format_exc())
    return None


def _findDimsCarQuery(cursor, car_make, car_year, car_model):
  ''' Infers the best matching entry in CarQueryDb and gets L, W, H, and wheelbase. '''

  # If the query car does not have a year.  
  if car_year is None:
    s = 'SELECT DISTINCT car_year FROM gt WHERE car_make=? AND car_model=? ORDER BY car_year ASC'
    _debug_execute(s, (car_make, car_model))
    cursor.execute(s, (car_make, car_model))
    result = cursor.fetchall()
    if result:
      logging.debug('The year is not known for %s, but there are models for years %s. Will take the max year.' % \
          (str((car_make, car_model)), str(result)))
      last_year = int(result[-1][0])
      s = 'SELECT DISTINCT dims_L, dims_W, dims_H, wheelbase FROM gt WHERE car_make=? AND car_year=? AND car_model=?'
      _debug_execute(s, (car_make, str(last_year), car_model))
      cursor.execute(s, (car_make, str(last_year), car_model))  # TODO: in v2 fix type of year.
      result = cursor.fetchone()
      assert result is not None
      return result
    else:
      logging.debug('Did not find info for any year for: %s' % str((car_make, car_model)))
      return None, None, None

  # The query car has the year field, try to match exactly.
  s = 'SELECT DISTINCT dims_L, dims_W, dims_H, wheelbase FROM gt WHERE car_make=? AND car_year=? AND car_model=?'
  _debug_execute(s, (car_make, car_year, car_model))
  cursor.execute(s, (car_make, car_year, car_model))
  result = cursor.fetchone()
  if result is not None:
    return result

  # Try other years.
  logging.debug('Did not find info for exactly: %s' % str((car_make, car_year, car_model)))
  s = 'SELECT DISTINCT car_year FROM gt WHERE car_make=? AND car_model=?'
  _debug_execute(s, (car_make, car_model))
  cursor.execute(s, (car_make, car_model))
  result = cursor.fetchall()
  if result:
    logging.debug('Did not find a model for year: %s, but there are models for years %s' % \
        (str((car_make, car_year, car_model)), str(result)))
    closest_year = min([x[0] for x in result], key=lambda x:abs(x - int(car_year)))  # TODO: in v2 fix type of year.
    s = 'SELECT DISTINCT dims_L, dims_W, dims_H, wheelbase FROM gt WHERE car_make=? AND car_year=? AND car_model=?'
    _debug_execute(s, (car_make, str(closest_year), car_model))
    cursor.execute(s, (car_make, str(closest_year), car_model))  # TODO: in v2 fix type of year.
    result = cursor.fetchone()
    assert result is not None
    return result

  logging.debug('Did not find info for any year for: %s' % str((car_make, car_model)))
  return None, None, None


def _findDimsInProperties(cursor, collection_id, model_id):
    # Length.
    s = 'SELECT label FROM clas WHERE collection_id=? AND model_id=? AND class="length"'
    _debug_execute(s, (collection_id, model_id))
    cursor.execute(s, (collection_id, model_id))
    length = cursor.fetchall()
    assert len(length) <= 1
    length = length[0] if len(length) == 1 else None
    # Width.
    s = 'SELECT label FROM clas WHERE collection_id=? AND model_id=? AND class="width"'
    _debug_execute(s, (collection_id, model_id))
    cursor.execute(s, (collection_id, model_id))
    width = cursor.fetchall()
    assert len(width) <= 1
    width = width[0] if len(width) == 1 else None
    # Heihgt.
    s = 'SELECT label FROM clas WHERE collection_id=? AND model_id=? AND class="height"'
    _debug_execute(s, (collection_id, model_id))
    cursor.execute(s, (collection_id, model_id))
    height = cursor.fetchall()
    assert len(height) <= 1
    height = height[0] if len(height) == 1 else None
    return length, width, height


def importCollectionsParser(subparsers):
  parser = subparsers.add_parser('importCollections',
    description='Import json file with the collection.')
  parser.add_argument('--collection_ids', nargs='+', required=True)
  parser.add_argument('--overwrite', action='store_true')
  parser.set_defaults(func=importCollections)

def importCollections(cursor, args):
  for collection_id in args.collection_ids:
    json_path = atcadillac('CAD/%s/collection.json' % collection_id)
    collection = json.load(open(json_path))
    logging.info('Found %d models in the collection' % len(collection['vehicles']))
    _importCollection(cursor, collection, args.overwrite)


def renderExamplesParser(subparsers):
  parser = subparsers.add_parser('renderExamples',
    description='Render an example for each model.')
  parser.add_argument('--overwrite', action='store_true')
  parser.set_defaults(func=renderExamples)

def renderExamples(cursor, args):
  for entry in _queryCollectionsAndModels(cursor, args.clause):
    _renderExample(entry[0], entry[1], args.overwrite)


def classifyParser(subparsers):
  parser = subparsers.add_parser('classify',
    description='Manually assign/change a property for each model.')
  parser.add_argument('--class_name', required=True)
  parser.add_argument('--key_dict_json', required=True,
      help='Which label in db each key will correspond. '
      'For class_name="issue" '
      '{"g": "matte glass", "t": "triangles", "c": "no color", "o": "other"}. '
      'For class_name="color" '
      '{"w": "white", "k": "black", "e": "gray", "r": "red", "y": "yellow", '
      '"g": "green", "b": "blue", "o": "orange"}. '
      'For class_name="domain" '
      '{"f": "fiction", "m": "military", "e": "emergency"}.'
      'For class_name="type1" '
      '{" ": "passenger", "t": "truck", "v": "van", "b": "bus", "c": "bike"}.')
  parser.set_defaults(func=classify)

def classify(cursor, args):

  # Parse a string into a dict.
  key_dict = json.loads(args.key_dict_json)
  logging.info('Key_dict:\n%s' % pformat(key_dict))

  # Remove those that cant be rendered.
  removeAllWithoutRender(cursor, args=argparse.Namespace(clause=''))

  entries = _queryCollectionsAndModels(cursor, args.clause)
  logging.info('Found %d model with .blend files.' % len(entries))
  if len(entries) == 0:
    return

  button = 0
  i = 0
  while True:

    # Going in a loop.
    if i == -1:
      logging.info('Looping to the last model.')
    if i == len(entries):
      logging.info('Looping to the first model.')
    i = i % len(entries)

    collection_id, model_id = entries[i]
    logging.info('i: %d model_id: %s, collection_id %s' % (i, model_id, collection_id))

    # Load the model's current label.
    cursor.execute('SELECT label FROM clas WHERE model_id=? AND collection_id=? AND class=?',
         (model_id, collection_id, args.class_name))
    labels = cursor.fetchall()
    if len(labels) == 0:
      label = None
    elif len(labels) == 1:
      label = labels[0][0]
    else:
      raise Exception('Too many labels for %s' % (model_id, collection_id, args.class_name))
    logging.debug('Current label is %s' % label)

    # Load the example image.
    example_path = getExamplePath(collection_id, model_id)
    assert op.exists(example_path), example_path
    image = cv2.imread(example_path)
    assert image is not None, example_path

    # Display
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0,255,255)
    thickness = 2
    cv2.putText (image, label, (50,50), font, 2, color, thickness)
    cv2.imshow('show', image)
    button = cv2.waitKey(-1)
    logging.debug('User pressed button: %d' % button)

    if button == 27:
      break
    elif button == ord('-'):
      logging.debug('Previous car.')
      i -= 1
    elif button == ord('='):
      logging.debug('Next car.')
      i += 1
    elif button == 127:  # Delete.
      logging.debug('Button "delete"')
      i += 1
      s = 'DELETE FROM clas WHERE class=? AND collection_id=? AND model_id=?'
      logging.debug('Will execute: %s' % s)
      cursor.execute(s, (args.class_name, collection_id, model_id))
    else:
      for key in key_dict:
        if button == ord(key):
          i += 1
          logging.debug('Button %s' % key)
          label = key_dict[key]

          # Update or insert.
          if len(labels) == 0:
            s = 'INSERT INTO clas(label,class,collection_id,model_id) VALUES (?,?,?,?)'
          elif len(labels) == 1:
            s = 'UPDATE clas SET label=? WHERE class=? AND collection_id=? AND model_id=?;'
          else:
            assert False
          logging.debug('Will execute: %s' % s)
          cursor.execute(s, (label, args.class_name, collection_id, model_id))
          break  # Key iteration.


def fillInDimsParser(subparsers):
  parser = subparsers.add_parser('fillInDims',
    description='Fill in the dimensions fields with actual dimensions.')
  parser.set_defaults(func=fillInDims)

def fillInDims(cursor, args):

  # Remove those that cant be rendered.
  removeAllWithoutRender(cursor, args=argparse.Namespace(clause=''))

  for idx, (collection_id, model_id) in enumerate(_queryCollectionsAndModels(cursor, args.clause)):
    logging.info('Model idx: %d' % idx)
    dims = _getDimsFromBlender(cursor, collection_id, model_id)
    if dims is None:
      continue
    dims_L, dims_W, dims_H = dims
    s = 'UPDATE cad SET dims_L=?, dims_W=?, dims_H=? WHERE collection_id=? AND model_id=?;'
    logging.debug('Will execute: %s' % s)
    cursor.execute(s, (dims_L, dims_W, dims_H, collection_id, model_id))


def removeAllWithoutBlendParser(subparsers):
  parser = subparsers.add_parser('removeAllWithoutBlend',
    description='Remove all models without the blend file.')
  parser.set_defaults(func=removeAllWithoutBlend)

def removeAllWithoutBlend (cursor, args):

  for collection_id, model_id in _queryCollectionsAndModels(cursor, args.clause):

    blend_path = getBlendPath(collection_id, model_id)
    if not op.exists(blend_path):
      logging.debug('Blend file does not exist: %s' % blend_path)

      s = 'DELETE FROM clas WHERE collection_id=? AND model_id=?'
      logging.debug('Will execute: %s' % s)
      cursor.execute(s, (collection_id, model_id))

      s = 'DELETE FROM cad WHERE collection_id=? AND model_id=?'
      logging.debug('Will execute: %s' % s)
      cursor.execute(s, (collection_id, model_id))

  cursor.execute('SELECT cad.collection_id, cad.model_id FROM cad %s;' % args.clause)
  entries = cursor.fetchall()
  logging.info('There are %d cad models with blend.' % len(entries))


def removeAllWithoutRenderParser(subparsers):
  parser = subparsers.add_parser('removeAllWithoutRender',
    description='Remove all models without rendered example as having an issue.')
  parser.set_defaults(func=removeAllWithoutRender)

def removeAllWithoutRender (cursor, args):

  for collection_id, model_id in _queryCollectionsAndModels(cursor, args.clause):

    example_path = getExamplePath(collection_id, model_id)
    if not op.exists(example_path):
      logging.debug('Example image does not exist: %s' % example_path)

      s = 'DELETE FROM clas WHERE collection_id=? AND model_id=?'
      logging.debug('Will execute: %s' % s)
      cursor.execute(s, (collection_id, model_id))

      s = 'DELETE FROM cad WHERE collection_id=? AND model_id=?'
      logging.debug('Will execute: %s' % s)
      cursor.execute(s, (collection_id, model_id))

  cursor.execute('SELECT cad.collection_id, cad.model_id FROM cad %s;' % args.clause)
  entries = cursor.fetchall()
  logging.info('There are %d cad models with render.' % len(entries))


def removeDuplicatesParser(subparsers):
  parser = subparsers.add_parser('removeDuplicates',
    description='Remove all models with duplicate model_id from other collections.')
  parser.set_defaults(func=removeDuplicates)

def removeDuplicates(cursor, args):

  s = 'SELECT COUNT(cad.model_id) - COUNT(DISTINCT(cad.model_id)) FROM cad %s;' % args.clause
  logging.debug('Will execute: %s' % s)
  cursor.execute(s)
  num_duplicates = cursor.fetchone()[0]
  logging.info('Found %d duplicates.' % num_duplicates)

  s = 'SELECT DISTINCT(cad.model_id) FROM cad %s;' % args.clause
  logging.debug('Will execute: %s' % s)
  cursor.execute(s)
  model_ids = cursor.fetchall()
  logging.info('Found %d distinct model ids' % len(model_ids))

  for model_id, in model_ids:

    s = 'SELECT collection_id FROM cad WHERE model_id=?;'
    cursor.execute(s, (model_id,))
    collection_ids = cursor.fetchall()

    # Not a duplicate, skip.
    if len(collection_ids) == 1:
      continue

    # Keep the first model.
    del collection_ids[0]

    # Remove the rest.
    for collection_id, in collection_ids:

      s = 'DELETE FROM clas WHERE collection_id=? AND model_id=?'
      logging.debug('Will execute: %s' % s)
      cursor.execute(s, (collection_id, model_id))

      s = 'DELETE FROM cad WHERE collection_id=? AND model_id=?'
      logging.debug('Will execute: %s' % s)
      cursor.execute(s, (collection_id, model_id))


def makeGridParser(subparsers):
  parser = subparsers.add_parser('makeGrid',
    description='Combine renders of models that satisfy a "where" into a grid.')
  parser.add_argument('--swidth', type=int, help='Width to crop. If not specified, use model-dependent crop.')
  parser.add_argument('--display', action='store_true')
  parser.add_argument('--at_most', type=int, help='if specified, return at most N models')
  parser.add_argument('--cols', type=int, default=4, help='number of columns.')
  parser.add_argument('--dwidth', type=int, default=512, help='output width of a cell')
  parser.add_argument('--dheight', type=int, default=384, help='output height of a cell')
  parser.add_argument('--out_path')
  parser.set_defaults(func=makeGrid)

def makeGrid (cursor, args):

  # Load empty image.
  if not args.swidth:
    empty_path = atcadillac('scenes/empty-import.png')
    if not op.exists(empty_path):
      raise Exception('Empty image does not exist at %s.' % empty_path)
    empty = cv2.imread(empty_path, -1)
    if empty is None:
      raise Exception('Failed to load empty image.')

  # Remove all models without render and duplicates.
  removeAllWithoutRender(cursor, args)
  removeDuplicates(cursor, args)

  entries = _queryCollectionsAndModels(cursor, args.clause)
  shuffle(entries)
  if args.at_most is not None and len(entries) > args.at_most:
    entries = entries[:args.at_most]

  if len(entries) == 0:
    logging.info('Nothing is found.')
    return

  rows = len(entries) // args.cols + (0 if len(entries) % args.cols == 0 else 1)
  logging.info('Grid is of element shape %d x %d' % (rows, args.cols))
  grid = np.zeros(shape=(rows * args.dheight, args.cols * args.dwidth, 4), dtype=np.uint8)
  logging.info('Grid is of pixel shape %d x %d' % (rows * args.dheight, args.cols * args.dwidth))

  for idx, (collection_id, model_id) in enumerate(entries):

    example_path = getExamplePath(collection_id, model_id)
    if not op.exists(example_path):
      logging.debug('Example image does not exist: %s' % example_path)

    render = cv2.imread(example_path, -1)
    if render is None:
      logging.error('Image %s failed to be read' % example_path)
      continue

    h_to_w = args.dheight / args.dwidth

    if args.swidth:  # Manually given crop.
      sheight = int(args.swidth * h_to_w)
      y1 = render.shape[0] // 2 - sheight // 2
      y2 = y1 + sheight
      x1 = render.shape[1] // 2 - args.swidth // 2
      x2 = x1 + args.swidth
      crop = render[y1:y2, x1:x2]

    else:  # Model-dependent crop.
      # Find tight crop.
      mask = (render - empty) != 0
      nonzeros = mask.nonzero()
      y1 = nonzeros[0].min()
      x1 = nonzeros[1].min()
      y2 = nonzeros[0].max()
      x2 = nonzeros[1].max()
      swidth = x2 - x1
      sheight = y2 - y1
      # Adjust to keep the ratio fixed.
      if sheight < h_to_w * swidth:
        sheight = int(swidth * h_to_w)
        y1 = int((y2 + y1) / 2 - sheight / 2)
        y2 = y1 + sheight
      else:
        swidth = int(sheight / h_to_w)
        x1 = int((x2 + x1) / 2 - swidth / 2)
        x2 = x1 + swidth
      logging.debug('Crop at: y1=%d, x1=%d, y2=%d, x2=%d.' % (y1, x1, y2, x2))
      if logging.getLogger().getEffectiveLevel() <= 10:
        diff = render - empty
        diff = cv2.rectangle(diff, (x1,y1), (x2,y2), (0,255,0), 1)
        cv2.imshow('diff', diff)
        cv2.waitKey(-1)
      # Add the padding in case the box went  and crop.
      H, W = render.shape[:2]
      render = np.pad(render, pad_width=((H,H),(W,W),(0,0)), mode='constant', constant_values=0)
      crop = render[y1+H : y2+H, x1+W : x2+W]

    crop = cv2.resize(crop, dsize=(args.dwidth, args.dheight))

    x1 = idx % args.cols * args.dwidth
    y1 = idx // args.cols * args.dheight
    logging.debug('Idx: %03d, x1: %05d, y1: %05d, collection_id: %s, model_id: %s' % 
        (idx, x1, y1, collection_id, model_id))
    grid[y1 : y1 + args.dheight, x1 : x1 + args.dwidth] = crop

  if args.display:
    cv2.imshow('grid', grid)
    cv2.waitKey(-1)

  if args.out_path:
    cv2.imwrite(args.out_path, grid)
    

def plotHistogramParser(subparsers):
  parser = subparsers.add_parser('plotHistogram',
    description='Get a 1d histogram plot of fields.')
  parser.set_defaults(func=plotHistogram)
  parser.add_argument('--query', required=True, help='e.g., SELECT car_make FROM cad.')
  parser.add_argument('--ylog', action='store_true')
  parser.add_argument('--bins', type=int)
  parser.add_argument('--xlabel', default='')
  parser.add_argument('--rotate_xticklabels', action='store_true')
  parser.add_argument('--categorical', action='store_true')
  parser.add_argument('--display', action='store_true', help='show on screen.')
  parser.add_argument('--show_unlabelled', action='store_true', help='include models without labels.')
  parser.add_argument('--out_path', help='if specified, will save the plot to this file.')

def plotHistogram(cursor, args):

  # Remove all models without render and duplicates.
  removeAllWithoutRender(cursor, args=argparse.Namespace(clause=''))
  removeDuplicates(cursor, args=argparse.Namespace(clause=''))

  cursor.execute(args.query)
  entries = cursor.fetchall()

  xlist = [x if x is not None else ('unlabelled' if args.show_unlabelled else None) for x, in entries]
  if not xlist:
    logging.info('No cars, nothing to draw.')
    return
  logging.debug(str(xlist))

  fig, ax = plt.subplots()
  if args.categorical:
    import pandas as pd
    import seaborn as sns
    if args.rotate_xticklabels:
      plt.xticks(rotation=90)
    data = pd.DataFrame({args.xlabel: xlist})
    ax = sns.countplot(x=args.xlabel, data=data, order=data[args.xlabel].value_counts().index)
    plt.tight_layout()
  else:
    if args.bins:
      ax.hist(xlist, args.bins)
    else:
      ax.hist(xlist)

  if args.ylog:
    ax.set_yscale('log', nonposy='clip')
  #plt.xlabel(args.x if args.xlabel else '')
  plt.ylabel('')
  if args.out_path:
    logging.info('Saving to %s' % args.out_path)
    plt.savefig(args.out_path)
  if args.display:
    plt.show()


def manuallyEditInBlenderParser(subparsers):
  parser = subparsers.add_parser('manuallyEditInBlender',
    description='For each model there is an option to edit and save its blender file.')
  parser.set_defaults(func=manuallyEditInBlender)

def manuallyEditInBlender(cursor, args):

  # Remove all models without render.
  removeAllWithoutRender(cursor, args=argparse.Namespace(clause=''))

  entries = _queryCollectionsAndModels(cursor, args.clause)

  button = 0
  i = 0
  while True:

    # Going in a loop.
    if i == -1:
      logging.info('Looping to the last model.')
    if i == len(entries):
      logging.info('Looping to the first model.')
    i = i % len(entries)

    collection_id, model_id = entries[i]
    logging.info('i: %d model_id: %s, collection_id %s' % (i, model_id, collection_id))

    # Load the example image.
    example_path = getExamplePath(collection_id, model_id)
    assert op.exists(example_path), example_path
    image = cv2.imread(example_path)
    assert image is not None, example_path

    cv2.imshow('show', image)
    button = cv2.waitKey(-1)
    logging.debug('User pressed button: %d' % button)

    if button == 27:
      break
    elif button == ord('-'):
      logging.debug('Previous car.')
      i -= 1
    elif button == ord('='):
      logging.debug('Next car.')
      i += 1
    elif button == 32:  # Space
      logging.debug('Button "space", will edit.')
      # i += 1
      try:
        command = ['%s/blender' % os.getenv('BLENDER_ROOT'),
                   getBlendPath(collection_id, model_id)]
        returncode = subprocess.call (command, shell=False)
        logging.debug('Blender returned code %s' % str(returncode))
        # Re-render.
        _renderExample (collection_id, model_id, overwrite=True)
      except:
        logging.error('Failed: %s' % traceback.format_exc())
        continue


def manuallyEditCarModelParser(subparsers):
  parser = subparsers.add_parser('manuallyEditCarModel',
    description='Edit car_model field manually.')
  parser.set_defaults(func=manuallyEditCarModel)
  parser.add_argument('--car_query_db_path', required=True)  # TODO: an option without it.
  parser.add_argument('--where', default='1', help='e.g., clause after WHERE.')
  parser.add_argument('--fields', default=['car_year', 'car_make', 'car_model'], nargs='+')

def manuallyEditCarModel(cursor, args):

  # Open CarQuery database.
  if not op.exists(args.car_query_db_path):
    raise Exception('Does not exist, create db with MakeCarQueryDb.py.')
  query_conn = sqlite3.connect(args.car_query_db_path)
  query_cursor = query_conn.cursor()

  if 'car_make' in args.fields:

    s = 'SELECT model_id, collection_id, model_name, description, car_make FROM cad WHERE (%s);' % args.where
    logging.debug('Will execute: %s' % s)
    cursor.execute(s)
    entries = cursor.fetchall()
    logging.info('Found %d entries' % len(entries))

    for model_id, collection_id, model_name, description, car_make in entries:

      query_cursor.execute('SELECT COUNT(1) FROM gt WHERE car_make=?', (car_make,))
      result = query_cursor.fetchone()[0]
      if result:
        logging.debug('Found the make "%s".\n' % car_make)
        continue

      query_cursor.execute('SELECT DISTINCT car_make FROM gt')
      result = query_cursor.fetchall()
      print('Available makes: \n%s' % pformat(sorted([x[0] for x in result]), indent=2))
      print('Could not find make "%s".' % car_make)
      print('Model_name:', model_name)
      print('Description:', description)
      print('Enter car_make. Empty input to skip, Space to exit:')
      car_make = input().lower()
      if car_make == '':
        continue
      if car_make == ' ':
        break
      s = 'UPDATE cad SET car_make=? WHERE model_id=? AND collection_id=?'
      cursor.execute(s, (car_make, model_id, collection_id))

  if 'car_model' in args.fields:

    s = 'SELECT model_id, collection_id, model_name, description, car_make, car_model FROM cad ' \
        'WHERE car_make IS NOT NULL AND (%s);' % args.where
    logging.debug('Will execute: %s' % s)
    cursor.execute(s)
    entries = cursor.fetchall()
    logging.info('Found %d entries' % len(entries))

    for model_id, collection_id, model_name, description, car_make, car_model in entries:

      s = 'SELECT COUNT(1) FROM gt WHERE car_make=? AND car_model=?'
      query_cursor.execute(s, (car_make, car_model))
      result = query_cursor.fetchone()[0]
      if result:
        logging.debug('Found the make "%s" and model "%s".\n' % (car_make, car_model))
        continue

      s = 'SELECT DISTINCT car_model FROM gt WHERE car_make=?'
      query_cursor.execute(s, (car_make,))
      result = query_cursor.fetchall()
      print('Available models: \n%s' % pformat(sorted([x[0] for x in result])  , indent=2))
      print('Could not find model "%s" for make "%s".' % (car_model, car_make))
      print('Model_name:', model_name)
      print('Description:', description)
      print('Enter car_model. Empty input to skip, Space to exit:')
      car_model = input().lower()
      if car_model == '':
        continue
      if car_model == ' ':
        break
      if car_model == 'null':
        s = 'UPDATE cad SET car_model=NULL WHERE model_id=? AND collection_id=?'
        cursor.execute(s, (model_id, collection_id))
        continue        
      s = 'UPDATE cad SET car_model=? WHERE model_id=? AND collection_id=?'
      cursor.execute(s, (car_model, model_id, collection_id))
  
  query_conn.close()


def fillDimsFromCarQueryDbParser(subparsers):
  parser = subparsers.add_parser('fillDimsFromCarQueryDb',
    description='For each model search the CarQueryDb for the dims info.')
  parser.set_defaults(func=fillDimsFromCarQueryDb)
  parser.add_argument('--car_query_db_path', required=True)

def fillDimsFromCarQueryDb(cursor, args):

  # Open CarQuery database.
  if not op.exists(args.car_query_db_path):
    raise Exception('Does not exist, create db with MakeCarQueryDb.py.')
  query_conn = sqlite3.connect(args.car_query_db_path)
  query_cursor = query_conn.cursor()

  s = 'SELECT collection_id, model_id, car_make, car_year, car_model FROM cad %s' % args.clause
  logging.debug('Will execute: %s' % s)
  cursor.execute(s)
  entries = cursor.fetchall()
  logging.info('Found %d entries' % len(entries))

  button = 0
  i = 0
  while True:

    if i == len(entries):
      logging.info('No more models.')
      break

    collection_id, model_id, car_make, car_year, car_model = entries[i]
    logging.info('i: %d model_id: %s, collection_id %s' % (i, model_id, collection_id))

    s = 'SELECT DISTINCT dims_L, dims_W, dims_H, wheelbase FROM gt WHERE car_make=? AND car_year=? AND car_model=?'
    result_carquery = _findDimsCarQuery(query_cursor, car_make, car_year, car_model)
    result_blender = _getDimsFromBlender(cursor, collection_id, model_id)
    logging.info('For collection_id: %s, model_id: %s:\n\tCar: "%s %s %s"\n\tBlender:  %s\n\tCarQuery: %s' %
        (collection_id, model_id, car_make, (car_year if car_year else ''), car_model,
         str(result_blender), str(result_carquery)))

    # Load the example image.
    example_path = getExamplePath(collection_id, model_id)
    assert op.exists(example_path), example_path
    image = cv2.imread(example_path)
    assert image is not None, example_path

    # Show image and ask for a key
    print ('Enter %s' % ','.join(['"%s"' % 'lwhb'[i] for i,x in enumerate(result_carquery) if x is not None]))
    s = 'SELECT comment FROM cad WHERE collection_id=? AND model_id=?'
    _debug_execute(s, (collection_id, model_id))
    cursor.execute(s, (collection_id, model_id))
    comment = cursor.fetchone()[0]
    if comment is not None:
      font = cv2.FONT_HERSHEY_SIMPLEX
      color = (0,255,255)
      thickness = 2
      logging.info('Comment: %s' % comment)
      cv2.putText (image, comment, (50,50), font, 2, color, thickness)
    cv2.imshow('show', image)
    button = cv2.waitKey(-1)
    logging.debug('User pressed button: %d' % button)

    if button == 27:
      break
    elif button == ord('-'):
      logging.debug('Previous car.')
      i -= 1
      continue
    elif button == ord('='):
      logging.debug('Next car.')
      i += 1
      continue
    elif button == ord(' '):
      logging.debug('User verified the model is correct.')
      i += 1
    elif button == ord('l') and result_carquery[0] is not None:
      scale = result_carquery[0] / result_blender[0]
      logging.debug('Button "l", will scale based on length, scale=%.3f.' % scale)
    elif button == ord('w') and result_carquery[1] is not None:
      scale = result_carquery[1] / result_blender[1]
      logging.debug('Button "w", will scale based on width, scale=%.3f.' % scale)
    elif button == ord('h') and result_carquery[2] is not None:
      scale = result_carquery[2] / result_blender[2]
      logging.debug('Button "h", will scale based on height, scale=%.3f.' % scale)
    elif button == ord('b') and result_carquery[3] is not None:
      scale = result_carquery[3] / result_blender[3]
      logging.debug('Button "b", will scale based on wheelbase, scale=%.3f.' % scale)

    elif button == ord('L'):
      logging.debug('Input the length in inches')
      inches = float(input())
      length = inches * 0.0254
      scale = length / result_blender[0]
    elif button == ord('W'):
      logging.debug('Input the width in inches')
      inches = float(input())
      width = inches * 0.0254
      scale = width / result_blender[1]
    elif button == ord('H'):
      logging.debug('Input the height in inches')
      inches = float(input())
      height = inches * 0.0254
      scale = height / result_blender[2]

    def _getS(axis):
      prop_dims = _findDimsInProperties(cursor, collection_id, model_id)
      if prop_dims[axis] is None:
        s = 'INSERT INTO clas(label,class,collection_id,model_id) VALUES (?,?,?,?)'
      else:
        s = 'UPDATE clas SET label=? WHERE class=? AND collection_id=? AND model_id=?'
      return s

    if button == ord('L'):
      s = _getS(0)
      _debug_execute(s, (length, 'length', collection_id, model_id))
      cursor.execute(s, (length, 'length', collection_id, model_id))
      length = None
    if button == ord('W'):
      s = _getS(1)
      _debug_execute(s, (width, 'width', collection_id, model_id))
      cursor.execute(s, (width, 'width', collection_id, model_id))
      width = None
    if button == ord('H'):
      s = _getS(2)
      _debug_execute(s, (height, 'height', collection_id, model_id))
      cursor.execute(s, (height, 'height', collection_id, model_id))
      height = None

    # Scale
    if button in [ord('l'), ord('w'), ord('h'), ord('b'), ord('L'), ord('W'), ord('H')]:
      model = {
          'blend_file': getBlendPath(collection_id, model_id),
          'scale': scale,
          'dry_run': 1 if args.dry_run else 0
      }
      model_path = op.join(COLLECTION_WORK_DIR, 'model.json')
      logging.info('Going to scale with scale == %.2f' % scale)
      with open(model_path, 'w') as f:
        f.write(json.dumps(model, indent=2))

      try:
        command = ['%s/blender' % os.getenv('BLENDER_ROOT'), '--background', '--python',
                   atcadillac('src/augmentation/collections/scale.py')]
        returncode = subprocess.call (command, shell=False)
        logging.info ('blender returned code %s' % str(returncode))
        # Re-render.
        _renderExample (collection_id, model_id, overwrite=True)
      except:
        logging.error('failed with exception: %s' % traceback.format_exc())
        break
      scale = None

    if button == ord('e'):
      try:
        command = ['%s/blender' % os.getenv('BLENDER_ROOT'),
                   getBlendPath(collection_id, model_id)]
        returncode = subprocess.call (command, shell=False)
        logging.debug('Blender returned code %s' % str(returncode))
        # Re-render.
        _renderExample (collection_id, model_id, overwrite=True)
      except:
        logging.error('failed with exception: %s' % traceback.format_exc())
        break

    # Update CAD.
    if not args.dry_run and button in [
        ord('l'), ord('w'), ord('h'), ord('b'), ord('e'), ord('L'), ord('H'), ord(' ')]:
      scale_by = 'user' if button == ord(' ') else chr(button)
      s = 'UPDATE cad SET comment="scale_by_%s" WHERE collection_id=? AND model_id=?' % scale_by
      _debug_execute(s, (collection_id, model_id))
      cursor.execute(s, (collection_id, model_id))

  query_conn.close()


if __name__ == "__main__":

  parser = argparse.ArgumentParser('Do one of the automatic operations on a db.')
  parser.add_argument('--in_db_file', required=True)
  parser.add_argument('--out_db_file', default=':memory:')
  parser.add_argument('--clause', default='', help='SQL WHERE clause.')
  parser.add_argument('--logging', type=int, default=20)
  parser.add_argument('--dry_run', action='store_true',
      help='Do not commit (for debugging).')

  subparsers = parser.add_subparsers()
  importCollectionsParser(subparsers)
  removeAllWithoutRenderParser(subparsers)
  removeAllWithoutBlendParser(subparsers)
  makeGridParser(subparsers)
  removeDuplicatesParser(subparsers)
  plotHistogramParser(subparsers)
  fillInDimsParser(subparsers)
  manuallyEditInBlenderParser(subparsers)
  renderExamplesParser(subparsers)
  classifyParser(subparsers)
  fillDimsFromCarQueryDbParser(subparsers)
  manuallyEditCarModelParser(subparsers)

  args = parser.parse_args()

  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

  # Backup the db.
  conn = safeConnect(args.in_db_file, args.out_db_file)
  cursor = conn.cursor()
  maybeCreateTableCad(cursor)
  maybeCreateTableClas(cursor)

  args.func(cursor, args)

  if not args.dry_run:
    logging.info('Committing changes.')
    conn.commit()
  conn.close()
