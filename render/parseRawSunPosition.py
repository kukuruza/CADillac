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
import os, os.path as op
import datetime
import pprint
import cPickle

from renderUtil import atcadillac

txt_file = 'resources/sun_position.txt'
bin_file = 'resources/sun_position.pkl'


with open(atcadillac(txt_file)) as f:
  lines = f.readlines()
  lines = lines[9:]

positions = {}

for line in lines:
  (daystr, clockstr, altitude, azimuth) = tuple(line.split())
  (month, day) = tuple(daystr.split('/'))
  (hour, minute) = tuple(clockstr.split(':'))

  date = datetime.datetime (year=2015, month=int(month), day=int(day), 
                            hour=int(hour), minute=int(minute))

  positions[date] = (float(altitude), float(azimuth))


with open(atcadillac(bin_file), 'wb') as f:
  cPickle.dump(positions, f, cPickle.HIGHEST_PROTOCOL)

# test

date = datetime.datetime (year=2015, month=6, day=14, hour=15, minute=45)

with open(atcadillac(bin_file), 'rb') as f:
  data = cPickle.load(f)
pprint.pprint(data[date])
