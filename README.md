# CADillac

A database of 3D car models and software to render the models on custom background using Blender.

The repo is in development.

## Dataset

The full set of 3D models can be downloaded from:

https://cmu.box.com/s/541gvrb7x4xkph8vi8xl15uqi8engfys

The dataset is organized as follows. `1be91e1e521b6756f500cc506a763c18` corresponds to the first out of several dozens of collections. It includes `blend` directory with Blender files, one per model, and `examples` directory with visualizations of these models. File `collections_v1.db` is the database with the information about each model.

```
CAD
collections_v1.db
|- 1be91e1e521b6756f500cc506a763c18
   |- blend
   |- examples
|- 5f08583b1f45a9a7c7193c87bbfa9088
   |- blend
   |- examples
...
```

## License

Apache 2

