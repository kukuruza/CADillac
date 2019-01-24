import csv
import numpy as np
import sqlite3
import logging
import argparse

from collectionUtilities import atcadillac, safeConnect
from collectionDb import maybeCreateTableCad, maybeCreateTableClas, getAllCadColumns

def maybeCreateTableGt(cursor):
  cursor.execute('''
      CREATE TABLE IF NOT EXISTS gt(
      idx INTEGER PRIMARY KEY,
      car_make TEXT,
      car_year INTEGER,
      car_model TEXT,
      car_trim TEXT,
      car_body TEXT,
      dims_L REAL,
      dims_W REAL,
      dims_H REAL,
      wheelbase REAL
  );''')
  cursor.execute('CREATE INDEX IF NOT EXISTS index_gt ON gt(car_make, car_year, car_model);')

def getAllGtColumns():
  return [
      'car_make',
      'car_year',
      'car_model',
      'car_trim',
      'car_body',
      'dims_L',
      'dims_W',
      'dims_H',
      'wheelbase',
  ]


def makeCarQueryDb(cursor):

  def convert(x, func=lambda x: x):
    ''' The CSV file has '' and 'NULL' for NULL. '''
    try:
      return None if x in ['', 'NULL', 'Not Available'] else func(x)
    except Exception:
      logging.error('Problem converting %s' % x)
      raise Exception()

  car_query_csv_path = atcadillac('resources/CQA_Advanced.csv')
  with open(car_query_csv_path, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    header = next(reader, None)  # Skip the header.

    for irow,row in enumerate(reader):

      make, year, model, trim, body, L, W, H, wheelbase = np.array(row)[[1,4,2,3,5,26,27,28,29],]
      try:
        make = convert(make, lambda x: x.lower())
        year = convert(year, lambda x: int(x))
        model = convert(model, lambda x: x.lower())
        L = convert(L, lambda x: float(x) / 1000)
        W = convert(W, lambda x: float(x) / 1000)
        H = convert(H, lambda x: float(x) / 1000)
        wheelbase = convert(wheelbase, lambda x: float(x) / 1000)
        trim = convert(trim, lambda x: x.lower())
        body = convert(body, lambda x: x.lower())
      except Exception as e:
        logging.error('Failed to parse row "%s" with exception: %s' %
            (np.array(row)[[1,4,2,3,5,26,27,28],], e))
        raise Exception()

      s = 'INSERT INTO gt(%s) VALUES (?,?,?,?,?,?,?,?,?)' % ','.join(getAllGtColumns())
      cursor.execute(s, (make, year, model, trim, body, L, W, H, wheelbase))

  for field in getAllGtColumns():
    cursor.execute('SELECT COUNT(1) FROM gt WHERE %s IS NOT NULL' % field)
    logging.info('Field %s has %d non-empty values.' % (field, cursor.fetchone()[0]))


def sanitizeCarQueryDb(cursor):

  def fixExtraSpaces(field):
    cursor.execute('SELECT COUNT(1) FROM gt WHERE %s != TRIM(%s)' % tuple([field] * 2))
    logging.info('Found %d "%s" fields with extra spaces' % (cursor.fetchone()[0], field))
    cursor.execute('UPDATE gt SET %s = TRIM(%s) WHERE %s != TRIM(%s)' % tuple([field] * 4))

  fixExtraSpaces('car_body')
  fixExtraSpaces('car_trim')
  fixExtraSpaces('car_make')
  fixExtraSpaces('car_model')


if __name__ == "__main__":

  parser = argparse.ArgumentParser('Make CarQuery Database that agree with collectionDb.')
  parser.add_argument('--in_db_file')
  parser.add_argument('--out_db_file', default=':memory:')
  args = parser.parse_args()

  logging.basicConfig(level=20, format='%(levelname)s: %(message)s')

  conn = safeConnect(args.in_db_file, args.out_db_file)
  cursor = conn.cursor()
  maybeCreateTableGt(cursor)

  makeCarQueryDb(cursor)
  sanitizeCarQueryDb(cursor)

  conn.commit()
  conn.close()
