import sys, os, os.path as op
import sqlite3
import logging
import argparse

sys.path.insert(0, op.dirname(op.dirname(os.path.realpath(__file__)))) # = '..'
from cads.collectionUtilities import safeConnect

if __name__ == "__main__":

  parser = argparse.ArgumentParser('Copies properties from CAD db to rendered db')
  parser.add_argument('--cad_db_path', required=True)
  parser.add_argument('--in_db_path', required=True)
  parser.add_argument('--out_db_path', required=True)
  parser.add_argument('--logging', type=int, default=20, choices=[10,20,30,40])
  parser.add_argument('--dry_run', action='store_true')
  args = parser.parse_args()

  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')
  #progressbar.streams.wrap_stderr()

  cad_conn = sqlite3.connect(args.cad_db_path)
  cad_cursor = cad_conn.cursor()

  conn = safeConnect(args.in_db_path, args.out_db_path)
  cursor = conn.cursor()

  cursor.execute('SELECT objectid,name FROM objects')
  object_entries = cursor.fetchall()
  for objectid,model_id, in object_entries:
    values = []
    cad_cursor.execute('SELECT class,label FROM clas WHERE model_id=?', (model_id,))
    entries = cad_cursor.fetchall()
    for key, value in entries:
      cursor.execute('INSERT INTO properties(objectid,key,value) VALUES (?,?,?)', (objectid,key,value))

  if not args.dry_run:
    conn.commit()
  conn.close()
  cad_conn.close()
