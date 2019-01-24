import sys, os, os.path as op
import logging
import simplejson as json
from pprint import pprint, pformat
import traceback
from datetime import datetime
import sqlite3


README_NAME = 'readme-blended.json'


class Cad:
  '''CAD collections'''

  def __init__(self, db_path):
    if not op.exists(db_path):
      raise Exception('Db does not exist at "%s"' % db_path)

    self.conn = sqlite3.connect(db_path)
    self.cursor = self.conn.cursor

  def get_model_by_id_and_collection (self, model_id, collection_id):
    self.cursor.execute('SELECT FROM ') 


  def get_ready_models (self, vehicle_type=None, collection_id=None):
    ''' get _source fields from all hits '''

    clauses = [self._is_ready(), self._is_valid()]
    if vehicle_type is not None:
      clauses.append(self._is_vehicle_type(vehicle_type))
    if collection_id is not None:
      clauses.append(self._is_in_collection(collection_id))

    body = {
      "size": 10000,
      "query": {
        "bool": {
          "must": clauses
        }
      }
    }
    logging.debug ('get_ready_models: %s' % pformat(body, indent=2))
    result = self.es.search(
        index=self.index_name,
        doc_type=self.type_name,
        body=body
    )
    hits = result['hits']['hits']
    logging.info ('get_ready_models: got %d hits' % len(hits))
    return [hit['_source'] for hit in hits]


  def get_random_ready_models (self, vehicle_type=None, collection_id=None, number=1):

    clauses = [self._is_ready(), self._is_valid()]
    if vehicle_type is not None:
      clauses.append(self._is_vehicle_type(vehicle_type))
    if collection_id is not None:
      clauses.append(self._is_in_collection(collection_id))

    body = {
      "size": number,
      "query": {
        "function_score": {
          "functions": [
            {
              "random_score" : {}
            }
          ],
          "score_mode": "sum",
          "query": {
            "bool": {
              "must": clauses
            }
          }
        }
      }
    }
    logging.debug ('get_random_ready_models: %s' % pformat(body, indent=2))
    result = self.es.search(
        index=self.index_name,
        doc_type=self.type_name,
        body=body)
    hits = result['hits']['hits']
    return [hit['_source'] for hit in hits]

