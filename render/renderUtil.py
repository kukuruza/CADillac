import os, os.path as op

def atcadillac (path):
  if path == ':memory:':
    return ':memory:'
  elif op.isabs(path):
    return path
  else:
    if not os.getenv('CADILLAC_DATA_PATH'):
        raise Exception ('Please set environmental variable CADILLAC_DATA_PATH')
    return op.join(os.getenv('CADILLAC_DATA_PATH'), path)

