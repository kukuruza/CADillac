#!/usr/bin/env python
from contextlib import closing
from selenium.webdriver import Firefox, FirefoxProfile
from selenium.webdriver.support.ui import WebDriverWait

import os, os.path as op
#import urllib2
import logging
import json
import string
import argparse
import shutil
import time
import traceback
from glob import glob

from collectionUtilities import atcadillac



CAD_DIR = atcadillac('CAD')
README_NAME = 'collection.json'


# delete special characters
def validateString (line):
    for char in '\n\t"\\':
      line = line.replace(char, '')
    return line


def _find_carmodel_in_vehicles_ (models_info, model_id):
    # TODO: replace sequential search with an elasticsearch index
    for carmodel in models_info:
        if carmodel['model_id'] == model_id:
            return carmodel
    return None



def download_model (browser, url, model_dir, args):

    # model_id is the last part of the url blahblah/model_id/model_name
    model_id = url.split('/')[-2]
    logging.info ('started with model_id: %s' % model_id)

    # open the page with model
    browser.get(url)
    WebDriverWait(browser, timeout=args.timeout).until(
        lambda x: x.find_element_by_id('model-title-banner'))

    # get the model name
    element = browser.find_element_by_id('model-title-banner')
    model_name = validateString(element.text.encode('ascii','ignore').decode("utf-8"))

    # get model description
    element = browser.find_element_by_class_name('description-text')
    description = validateString(element.text.encode('ascii','ignore').decode("utf-8"))

    model_info = {'model_id':     model_id,
                  'model_name':   model_name,
                  'description':  description,
                 }
    logging.debug ('vehicle info: %s' % str(model_info))
    print (json.dumps(model_info, indent=4))

    if not args.only_info:

      skp_path = op.join(model_dir, '%s.skp' % model_id)
      if op.exists(skp_path): 
          logging.info ('skp file already exists for model_id: %s' % model_id)
          return model_info

      # click on download button
      button = browser.find_element_by_class_name('button.button-download.nonSkpClient')
      button.click()

      # wait for the page to load the download buttons
      #   we check the latest skp versin first, then next, and next. Then raise.
      for skp_version in ['s15', 's14', 's13', None]:
          # skp_version None is the marker to give up
          if skp_version is None:
              raise Exception('Cannot find skp versions 15, 14, or 13')
          try:
              logging.info('trying to download skp version %s' % skp_version)
              WebDriverWait(browser, timeout=args.timeout).until(
                  lambda x: x.find_element_by_id('download-option-%s' % skp_version))
              break
          except:
              logging.info('model has not skp version %s. Try next.' % skp_version)


      # press model download button.
      button = browser.find_element_by_id('download-option-%s' % skp_version)
      button.click()

      # Changing the name of the file.
      have_moved = False
      confirmed = False
      logging.info('waiting for download to finish.')
      for itry in range(120):  # 2 minute.
        logging.debug('see if completed, try %d.' % itry)

        # look for Confirm button.
        if not confirmed:
          try:
            WebDriverWait(browser, timeout=1).until(
                lambda x: x.find_element_by_class_name('modal-dialog-button-ok'))
            button = browser.find_element_by_class_name('modal-dialog-button-ok')
            button.click()
            confirmed = True
          except:
            logging.info('was not ask to agree to conditions.')

        else:
          time.sleep(1)

        tmp_skp_paths = glob(op.join(model_dir, 'tmp', '*.skp'))
        assert len(tmp_skp_paths) <= 1, tmp_skp_paths
        if len(tmp_skp_paths) == 1:
          tmp_skp_path = tmp_skp_paths[0]
          if os.stat(tmp_skp_path).st_size == 0:
            logging.warning('file is still zero bytes.')
          else:
            shutil.move(tmp_skp_path, skp_path)
            try:
              logging.info('have renamed %s to %s' % 
                  (op.basename(tmp_skp_path), op.basename(skp_path)))
            except:
              logging.warning('cant print what was renamed because of non ascii.')
            have_moved = True
            break
      assert have_moved

      # download the model
#      logging.info ('downloading model_id: %s' % model_id)
#      logging.debug('downloading skp from url: %s' % skp_href)
#      f = urllib2.urlopen(skp_href)
#      with open(skp_path, 'wb') as local_file:
#          local_file.write(f.read())

    logging.info ('finished with model_id: %s' % model_id)
    return model_info



def download_all_models (browser, model_urls, models_info, collection_id, collection_dir):

    new_models_info = []
    counts = {'skipped': 0, 'downloaded': 0, 'failed': 0}

    # got to each model and download it
    for model_url in model_urls:
        model_id = model_url.split('=')[-1]
        model_info = _find_carmodel_in_vehicles_ (models_info, model_id)

        # if this model was previously failed to be recorded
        if model_info is not None:
            if model_info['error'] is not None:
                assert 'error' in model_info
                if model_info['error'] == 'download failed: timeout error':
                    logging.info ('re-doing previously failed download: %s' % model_id)
                else:
                    logging.info ('skipping bad for some reason model: %s' % model_id)
                    counts['skipped'] += 1
                    new_models_info.append(model_info)
                    continue
            else:
                logging.info ('skipping previously downloaded model_id %s' % model_id)
                counts['skipped'] += 1
                new_models_info.append(model_info)
                continue

        # # check if this model is known as a part of some other collection
        # seen_collection_ids = cad.is_model_in_other_collections (model_id, collection_id)
        # if seen_collection_ids:
        #     error = 'is a part of %d collections. First is %s' % \
        #                  (len(seen_collection_ids), seen_collection_ids[0])
        #     model_info = {'model_id': model_id, 'error': error}
        #     counts['skipped'] += 1
        #     logging.warning ('model_id %s %s' % (model_id, error))
        #     new_models_info.append(model_info)
        #     continue

        # process the model
        try:
            logging.debug('model url: %s' % model_url)
            model_dir = op.join(collection_dir, 'skp')
            model_info = download_model (browser, model_url, model_dir, args)
            counts['downloaded'] += 1
        except:
            logging.error('model_id %s was not downloaded because of error: %s'
               % (model_id, traceback.format_exc()))
            model_info = {'model_id': model_id, 
                          'error': 'download failed: timeout error'}
            counts['failed'] += 1
            continue

        new_models_info.append(model_info)

        # backing up on each step
        with open (op.join(collection_dir, README_NAME + '-models.backup'), 'w') as f:
            f.write(json.dumps(new_models_info, indent=4))


    logging.info ('out of %d models in collection: \n' % len(model_urls) +
                  '    skipped:     %d\n' % counts['skipped'] + 
                  '    downloaded:  %d\n' % counts['downloaded'] +
                  '    failed:      %d\n' % counts['failed'])

    return new_models_info    



def download_collection (collection_id, args):

  # collection_id is the last part of the url
  url = 'https://3dwarehouse.sketchup.com/collection.html?id=%s' % collection_id
  collection_dir = op.join(CAD_DIR, collection_id)
  logging.info ('will download coleection_id: %s' % collection_id)

  # if collection exists
  collection_path = op.join(collection_dir, README_NAME)
  if op.exists(collection_path):
      # if 'overwrite' enabled, remove everything and write from scratch
      if args.overwrite_collection:
          shutil.rmtree(collection_dir)
          models_info = []
      else:
          # if 'overwrite' disabled, try to read what was downloaded
          try:
              collection_info = json.load(open(collection_path))
              models_info = collection_info['vehicles']
              if models_info is None:
                models_info = []
          # if 'overwrite' disabled and can't read/parse the readme
          except:
              raise Exception('Failed to parse the collection due to: %s'
                  % sys.exc_info()[0])
  else:
      models_info = []
      if not op.exists(op.join(collection_dir, 'skp')):
          os.makedirs(op.join(collection_dir, 'skp'))

  profile = FirefoxProfile()
  profile.set_preference("browser.download.folderList", 2)
  profile.set_preference("browser.download.manager.showWhenStarting", False)
  profile.set_preference("browser.download.dir", op.join(collection_dir, 'skp', 'tmp'))
  if not op.exists(op.join(collection_dir, 'skp', 'tmp')):
    os.makedirs(op.join(collection_dir, 'skp', 'tmp'))
  profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/vnd.koan")

  with closing(Firefox(profile)) as browser:

    # open the page with collection
    browser.get(url)
    WebDriverWait(browser, timeout=args.timeout).until(
        lambda x: x.find_elements_by_class_name('results-entity-link'))

    # get collection name
    element = browser.find_element_by_id('title')
    print (element.text.encode('ascii', 'ignore').decode("utf-8"))
    collection_name = validateString(element.text.encode('ascii', 'ignore').decode("utf-8"))

    # get collection description
    element = browser.find_element_by_id('description')
    collection_description = validateString(element.text.encode('ascii','ignore').decode("utf-8"))

    # get collection tags
    #element = browser.find_element_by_id('tags')
    #element.find_element_by_xpath(".//p[@id='test']").text 
    #collection_name = element.text.encode('ascii','ignore').decode("utf-8")
    #collection_name = validateString(collection_name)

    # get author
    element = browser.find_element_by_id('collection-author')
    author_href = element.get_attribute('href')
    author_id = author_href.split('=')[-1]
    author_name = validateString(element.text.encode('ascii','ignore').decode("utf-8"))

    # keep scrolling the page until models show up (for pages with many models)
    prev_number = 0
    while True:
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        elements = browser.find_elements_by_class_name('results-entity-link')
        logging.info ('found %d models' % len(elements))
        if prev_number == len(elements):
            break
        else:
            prev_number = len(elements)
            time.sleep(1)
    # get the model urls
    model_urls = []
    for element in elements:
        model_url = element.get_attribute('href')
        model_urls.append(model_url)

    # download all models
    new_models_info = download_all_models (browser, model_urls, models_info, 
                                           collection_id, collection_dir)

    collection_info = {'collection_id': collection_id,
                       'collection_name': collection_name,
                       'author_id': author_id,
                       'author_name': author_name,
                       'vehicles': new_models_info
                       }

    with open (op.join(collection_dir, README_NAME), 'w') as f:
        f.write(json.dumps(collection_info, indent=4))



def download_author_models (author_id, args):
  ''' Write models of an author, which are not in any collection '''
  
  with closing(Firefox()) as browser:

    # collection_id is made up as 'author-%s' % author_id
    url = 'https://3dwarehouse.sketchup.com/user.html?id=%s' % author_id
    collection_id = 'author-%s' % author_id
    collection_dir = op.join(CAD_DIR, collection_id)
    logging.info ('will download coleection_id: %s' % collection_id)

    # if collection exists
    collection_path = op.join(collection_dir, README_NAME)
    if op.exists(collection_path):
        # if 'overwrite' enabled, remove everything and write from scratch
        if args.overwrite_collection:
            shutil.rmtree(collection_dir)
        else:
            # if 'overwrite' disabled, try to read what was downloaded
            try:
                collection_info = json.load(open(collection_path))
                models_info = collection_info['vehicles']
            # if 'overwrite' disabled and can't read/parse the readme
            except:
                raise Exception('Failed to parse the collection due to: %s'
                    % sys.exc_info()[0])
    else:
        models_info = []
        if not op.exists(op.join(collection_dir, 'skp')):
            os.makedirs(op.join(collection_dir, 'skp'))

    # open the page with collection
    browser.get(url)
    WebDriverWait(browser, timeout=args.timeout).until(
        lambda x: x.find_elements_by_class_name('results-entity-link'))

    # get author
    element = browser.find_element_by_id('display-name')
    author_name = validateString(element.text.encode('ascii','ignore').decode("utf-8"))

    # keep scrolling the page until models show up (for pages with many models)
    prev_number = 0
    while True:
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        elements = browser.find_elements_by_class_name('results-entity-link')
        logging.info ('found %d models' % len(elements))
        if prev_number == len(elements):
            break
        else:
            prev_number = len(elements)
            time.sleep(1)
    # get the model urls
    model_urls = []
    for element in elements:
        model_url = element.get_attribute('href')
        model_urls.append(model_url)

    # download all models
    new_models_info = download_all_models (model_urls, models_info, 
                                           collection_id, collection_dir)

    collection_info = {'collection_id': collection_id,
                       'collection_name': '',
                       'author_id': author_id,
                       'author_name': author_name,
                       'vehicles': new_models_info
                       }

    with open (op.join(collection_dir, README_NAME), 'w') as f:
        f.write(json.dumps(collection_info, indent=4))




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--collection_id')
    group.add_argument('--author_id')
    parser.add_argument('--overwrite_collection', action='store_true')
    parser.add_argument('--only_info', action='store_true', help='Not download any skp.')
    parser.add_argument('--timeout', default=10, type=int)
    parser.add_argument('--logging', type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

    # use firefox to get page with javascript generated content
    if args.collection_id is not None:
        download_collection (args.collection_id, args)
    elif args.author_id is not None:
        download_author_models (args.author_id, args)
