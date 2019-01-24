#!/usr/bin/env python
import os, os.path as op
import argparse
import logging
import sqlite3

from collectionUtilities import atcadillac, safeCopy
from collectionDb import maybeCreateTableCad


def copyDataBetweenDbs(cursor_in, cursor_out, fields):

  # Get data from input
  sin = 'SELECT %s,model_id,collection_id FROM cad' % ','.join(fields)
  logging.debug('Will execute on input: %s' % sin)
  cursor_in.execute(sin)

  # Query to execute on output for each data row.
  fstr = ', '.join(['%s=?' % f for f in fields])
  sout = 'UPDATE cad SET %s WHERE model_id=? AND collection_id=?' % fstr
  logging.debug('Will execute on output: %s' % sout)

  for entry in cursor_in.fetchall():
    cursor_out.execute(sout, entry)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--in_db_file', required=True)
  parser.add_argument('--out_db_file', default=':memory:')
  parser.add_argument('--logging', type=int, default=20)
  parser.add_argument('--fields', required=True, nargs='+')
  args = parser.parse_args()

  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

  # Backup the output db.
  safeCopy(args.out_db_file, args.out_db_file)

  conn_in = sqlite3.connect(atcadillac(args.in_db_file))
  cursor_in = conn_in.cursor()

  conn_out = sqlite3.connect(atcadillac(args.out_db_file))
  cursor_out = conn_out.cursor()
  maybeCreateTableCad(cursor_out)  # In case of in-memory db.

  copyDataBetweenDbs(cursor_in, cursor_out, args.fields)

  conn_out.commit()
  conn_out.close()
  conn_in.close()
