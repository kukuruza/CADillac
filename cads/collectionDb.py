import sqlite3


def maybeCreateTableCad(cursor):  # TODO: in v2 fix type of year.
  cursor.execute('''
      CREATE TABLE IF NOT EXISTS cad(
      model_id TEXT,
      collection_id TEXT,
      model_name TEXT,
      description TEXT, 
      error TEXT,
      car_make TEXT,
      car_year TEXT,
      car_model TEXT,
      color TEXT,
      dims_L REAL,
      dims_W REAL,
      dims_H REAL,
      comment TEXT,
      PRIMARY KEY (model_id, collection_id)
  );''')
  cursor.execute('CREATE INDEX IF NOT EXISTS cad_modelid ON cad(model_id);')
  cursor.execute('CREATE INDEX IF NOT EXISTS cad_modelid_collection ON cad(model_id, collection_id);')

def getAllCadColumns():
  return [
      'model_id',
      'collection_id',
      'model_name',
      'description', 
      'error',
      'car_make',
      'car_year',
      'car_model',
      'color',
      'dims_L',
      'dims_W',
      'dims_H',
      'comment'
  ]

def maybeCreateTableClas(cursor):
  cursor.execute('''
      CREATE TABLE IF NOT EXISTS clas(
      class TEXT,
      model_id TEXT,
      collection_id TEXT,
      label TEXT,
      comment TEXT,
      PRIMARY KEY (class, model_id, collection_id)
  );''')
  cursor.execute('CREATE INDEX IF NOT EXISTS clas_class ON clas(class);')
  cursor.execute('CREATE INDEX IF NOT EXISTS clas_modelid ON clas(model_id);')
  cursor.execute('CREATE INDEX IF NOT EXISTS clas_modelid_collection ON clas(model_id, collection_id);')
