```
export CADILLAC_DATA_PATH=$CITY_PATH/data/augmentation
PATCHDIR=$CADILLAC_DATA_PATH/test
```

Augmentation
============

## To make patches:

```
# Render.
python3 render/MakePatches.py \
  -o $PATCHDIR/scenes.db \
  --num_sessions 10 --num_per_session 3 --num_occluding 1 2 3 4 5 --mode PARALLEL \
  --clause_main 'WHERE error IS NULL AND dims_L < 10' \
  --cad_db_path $CADILLAC_DATA_PATH/CAD/collections_v1.db

# Copy CAD info.
python3 render/CopyCadProperties.py \
  --in_db_path  $PATCHDIR/scenes.db \
  --out_db_path $PATCHDIR/scenes-filled.db \
  --cad_db_path $CADILLAC_DATA_PATH/CAD/collections_v1.db

# Crop.
python3 ~/projects/shuffler/shuffler.py \
  --rootdir $PATCHDIR \
  -i $PATCHDIR/scenes-filled.db \
  -o $PATCHDIR/patches-e02-w64.db \
  filterObjectsAtBorder   \| \
  expandBoxes --expand_perc 0.2   \| \
  cropObjects --edges distort --target_width 64 --target_height 64 --media video  \
    --image_path $PATCHDIR/patches-e02-w64.avi \
    --mask_path $PATCHDIR/patches-e02-w64mask.avi

# Write how much background is seen.
python3 render/RecordInfiniteBack.py \
  --in_db_path  $PATCHDIR/patches-e02-w64.db \
  --out_db_path $PATCHDIR/patches-e02-w64.db \
  --rootdir $CADILLAC_DATA_PATH/test

# Hide background from mask.
python3 ~/projects/shuffler/shuffler.py \
  --rootdir $PATCHDIR \
  -i $PATCHDIR/patches-e02-w64.db \
  -o $PATCHDIR/patches-e02-w64-repainted.db \
  repaintMask --media video \
    --mask_path $PATCHDIR/patches-e02-w64mask-noback.avi \
    --mask_mapping_dict "{255: 255, 0: 0, 128: 0}" --overwrite
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
