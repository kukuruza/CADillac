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
import sqlite3
import logging
import argparse
import numpy as np

sys.path.insert(0, op.dirname(op.dirname(os.path.realpath(__file__)))) # = '..'
from cads.collectionUtilities import safeConnect

sys.path.insert(0, op.join(os.getenv('SHUFFLER_PATH'), 'lib'))
from backendMedia import MediaReader


if __name__ == "__main__":

  parser = argparse.ArgumentParser('Compute how much of the area is masked as bad.')
  parser.add_argument('--in_db_path', required=True)
  parser.add_argument('--out_db_path', required=True)
  parser.add_argument('--rootdir', required=True)
  parser.add_argument('--logging', type=int, default=20, choices=[10,20,30,40])
  parser.add_argument('--dry_run', action='store_true')
  args = parser.parse_args()

  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

  conn = safeConnect(args.in_db_path, args.out_db_path)
  cursor = conn.cursor()

  imreader = MediaReader(rootdir=args.rootdir)

  cursor.execute('SELECT maskfile,objectid FROM objects o LEFT JOIN images i ON o.imagefile = i.imagefile')
  entries = cursor.fetchall()
  for maskfile,objectid in entries:
    mask = imreader.imread(maskfile)
    percent = np.count_nonzero(mask == 128) / float(mask.size)
    cursor.execute('INSERT INTO properties(objectid,key,value) VALUES (?,?,?)',
      (objectid, 'background', '%.2f' % percent))

  if not args.dry_run:
    conn.commit()
  conn.close()
