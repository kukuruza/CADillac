```
export CADILLAC_DATA_PATH=$CITY_PATH/data/augmentation
```

Augmentation
============

## To make patches:

```
python3 render/MakePatches.py \
  -o $CADILLAC_DATA_PATH/test/scenes.db \
  --num_sessions 10 --num_per_session 10 --num_occluding 5 --mode PARALLEL \
  --clause_main 'WHERE error IS NULL AND dims_L < 10' \
  --cad_db_path $CADILLAC_DATA_PATH/CAD/collections_v1.db

python3 render/CopyCadProperties.py \
  --in_db_path  $CADILLAC_DATA_PATH/test/scenes.db \
  --out_db_path $CADILLAC_DATA_PATH/test/scenes-filled.db \
  --cad_db_path $CADILLAC_DATA_PATH/CAD/collections_v1.db

python3 ~/projects/shuffler/shuffler.py \
  --root $CADILLAC_DATA_PATH/test \
  -i $CADILLAC_DATA_PATH/test/scenes-filled.db \
  -o $CADILLAC_DATA_PATH/test/patches-w55-e04.db \
  filterObjectsAtBorder   \| \
  expandBoxes --expand_perc 0.2   \| \
  cropObjects --edges distort --target_width 64 --target_height 64 --media video  \
    --image_path $CADILLAC_DATA_PATH/test/patches-w55-e04.avi \
    --mask_path $CADILLAC_DATA_PATH/test/patches-w55-e04mask.avi
```

```
python src/augmentation/ProcessFrame.py --video_dir augmentation/scenes/cam572/Nov28-10h
```

```
python src/augmentation/GenerateTraffic.py  --job_file augmentation/jobs/572-Feb23-09h-test.json --traffic_file augmentation/video/test/traffic.json
```

```
python src/augmentation/ProcessVideo.py --timeout 10 --job_file augmentation/jobs/572-Feb23-09h-test.json --traffic_file augmentation/video/test/traffic.json
```
