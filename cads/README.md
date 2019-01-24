```
export CADILLAC_DATA_PATH=$CITY_PATH/data/augmentation
```


Scripts:

```
collection_id=\'5f08583b1f45a9a7c7193c87bbfa9088\'  # Quotes are important in "clause" arg.

# Import collections.
python3 cads/Modify.py \
  --in_db_file $CADILLAC_DATA_PATH/CAD/collections_v1.db \
  importCollections \
  --collection_ids ${collection_id}

# Classify color.
python3 cads/Modify.py \
  --in_db_file $CADILLAC_DATA_PATH/CAD/collections.db \
  --out_db_file $CADILLAC_DATA_PATH/CAD/collections.db \
  --clause "collection_id=${collection_id}" \
  --class_name=color --key_dict_json='{"w": "white", "k": "black", "e": "gray", "r": "red", "y": "yellow", "g": "green", "b": "blue", "o": "orange"}'

# Correct model_name.
python3 cads/Modify.py \
  --in_db_file $CADILLAC_DATA_PATH/CAD/collections_v1.db \
  manuallyEditCarModel \
  --car_query_db_path $CADILLAC_DATA_PATH/resources/CQA_Advanced_v1.db
```

SQL queries:

```
# Display the number of cars of each color.
SELECT label, COUNT(1) FROM clas WHERE class='color' GROUP BY label ORDER BY COUNT(1) DESC

# Display the number in each collection.
SELECT collection_id, COUNT(1) FROM cad GROUP BY collection_id ORDER BY COUNT(1) DESC

# Display car_make with its count.
SELECT car_make, COUNT(1) FROM cad GROUP BY car_ma ORDER BY COUNT(1) DESC

# Copy issue to error field.
UPDATE cad SET error = (SELECT clas.label FROM clas WHERE clas.model_id == cad.model_id AND clas.collection_id == cad.collection_id AND clas.class == 'issue') WHERE EXISTS (SELECT * FROM clas WHERE clas.model_id == cad.model_id AND clas.collection_id == cad.collection_id AND clas.class == 'issue')
```


Visualization:
```
# White Ford.
python3 cads/Modify.py \
  --in_db_file $CADILLAC_DATA_PATH/CAD/collections_v1.db \
  --clause 'INNER JOIN clas c1 ON cad.model_id=c1.model_id WHERE c1.label = "white" AND cad.car_make == "ford" AND error IS NULL' \
  makeGrid \
  --out_path $CADILLAC_DATA_PATH/CAD/-visualizations/v1_white_ford.png

# Van.
python3 cads/Modify.py \
  --in_db_file $CADILLAC_DATA_PATH/CAD/collections.db \
  --clause 'INNER JOIN clas ON cad.model_id=clas.model_id WHERE clas.label = "van" AND cad.model_id NOT IN (SELECT model_id FROM clas WHERE class == "issue")' \
  makeGrid \
  --out_path $CADILLAC_DATA_PATH/CAD/-visualizations/v1_van.png

# Toyota truck.
python3 cads/Modify.py \
  --in_db_file $CADILLAC_DATA_PATH/CAD/collections_v1.db \
  --clause 'INNER JOIN clas ON cad.model_id=clas.model_id WHERE cad.car_make == "toyota" AND clas.label = "truck" AND error IS NULL' \
  makeGrid \
  --out_path $CADILLAC_DATA_PATH/CAD/-visualizations/v1_toyota_truck.png

# Military.
python3 cads/Modify.py \
  --in_db_file $CADILLAC_DATA_PATH/CAD/collections.db \
  --clause 'INNER JOIN clas ON cad.model_id=clas.model_id WHERE clas.label = "military" AND cad.model_id NOT IN (SELECT model_id FROM clas WHERE class == "issue")' \
  makeGrid \
  --out_path $CADILLAC_DATA_PATH/CAD/-visualizations/v1_military.png

# Fiction.
python3 cads/Modify.py \
  --in_db_file $CADILLAC_DATA_PATH/CAD/collections.db \
  --clause 'INNER JOIN clas ON cad.model_id=clas.model_id WHERE clas.label = "fiction" AND cad.model_id NOT IN (SELECT model_id FROM clas WHERE class == "issue")' \
  makeGrid \
  --out_path $CADILLAC_DATA_PATH/CAD/-visualizations/v1_fiction.png

# Cars longer than X1 and shorter than X2 (on collecton_v1).
python3 cads/Modify.py \
  --in_db_file $CADILLAC_DATA_PATH/CAD/collections_v1.db \
  --clause 'WHERE dims_L >= 9 AND dims_L <= 10 AND error ISNULL' \
  makeGrid \
  --out_path $CADILLAC_DATA_PATH/CAD/-visualizations/v1_length_9_to_10.png \
  --at_most 8

% Error: matte glass
python3 cads/Modify.py \
  --in_db_file $CADILLAC_DATA_PATH/CAD/collections_v1.db \
  --clause 'WHERE error == "matte glass"' \
  makeGrid \
  --out_path $CADILLAC_DATA_PATH/CAD/-visualizations/v1_error_matte_glass.png \
  --at_most 8

% Error: triangles
python3 cads/Modify.py \
  --in_db_file $CADILLAC_DATA_PATH/CAD/collections_v1.db \
  --clause 'WHERE error == "triangles"' \
  makeGrid \
  --out_path $CADILLAC_DATA_PATH/CAD/-visualizations/v1_error_triangles.png \
  --at_most 8

# Histogram of lengths (on collection_v1).
python3 cads/Modify.py \
  --in_db_file $CADILLAC_DATA_PATH/CAD/collections_v1.db \
  plotHistogram \
  --query 'SELECT dims_L FROM cad WHERE error ISNULL AND dims_L <= 25' \
  --xlabel 'length, m' --ylog \
  --out_path $CADILLAC_DATA_PATH/CAD/-visualizations/v1_hist_length.eps

# Histogram of car makes which have at least 5 models.
python3 cads/Modify.py \
  --in_db_file $CADILLAC_DATA_PATH/CAD/collections.db \
  plotHistogram \
  --query 'SELECT car_make FROM cad WHERE car_make IN (SELECT car_make FROM cad GROUP BY car_make HAVING COUNT(car_make) >= 5) AND model_id NOT IN (SELECT model_id FROM clas WHERE class == "issue")' \
  --categorical \
  --rotate_xticklabels \
  --out_path $CADILLAC_DATA_PATH/CAD/-visualizations/v1_hist_make_ge5.eps

# Histogram of car types1.
python3 cads/Modify.py \
  --in_db_file $CADILLAC_DATA_PATH/CAD/collections.db \
  plotHistogram \
  --query 'SELECT label FROM clas WHERE class="type1" AND model_id NOT IN (SELECT model_id FROM clas WHERE class == "issue")' \
  --categorical \
  --out_path $CADILLAC_DATA_PATH/CAD/-visualizations/v1_hist_type1.eps
```


## Work with CarQueryDb

```
# Create a db.
python3 cads/MakeCarQueryDb.py \
  --out_db_file $CADILLAC_DATA_PATH/resources/CQA_Advanced_v1.1.db

# Look up how many models are in this CarQueryDb.
python3 cads/Modify.py \
  --in_db_file $CADILLAC_DATA_PATH/CAD/collections_v1.db \
  --out_db_file $CADILLAC_DATA_PATH/CAD/collections_v1.db \
  --clause 'WHERE car_make IS NOT NULL AND car_model IS NOT NULL AND comment IS NULL' \
  fillDimsFromCarQueryDb \
  --car_query_db_path $CADILLAC_DATA_PATH/resources/CQA_Advanced_v1.1.db
```
