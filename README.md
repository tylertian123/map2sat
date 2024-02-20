# APS360
APS360 Project

Run the `dataset_generation/gen_dataset.py` script to generate datasets.

```
usage: gen_dataset.py [-h] [--rotation ROTATION] [--map-dir MAP_DIR] [--sat-dir SAT_DIR] [--zoom ZOOM] [--bearing BEARING] [--img-width IMG_WIDTH] [--img-height IMG_HEIGHT] [--workers WORKERS]
                      lat lon width height h_samples v_samples

positional arguments:
  lat                   Latitude of the top-left corner of the ROI.
  lon                   Longitude of the top-left corner of the ROI.
  width                 Width of the ROI in km.
  height                Height of the ROI in km.
  h_samples             Sample grid size (horizontal).
  v_samples             Sample grid size (vertical).

options:
  -h, --help            show this help message and exit
  --rotation ROTATION   How much the ROI rectangle is rotated (from being aligned with lat/lon lines) in degrees. Positive rotation is counterclockwise.
  --map-dir MAP_DIR     Directory to store the map images.
  --sat-dir SAT_DIR     Directory to store the satellite images.
  --zoom ZOOM           Zoom level.
  --bearing BEARING     Map rotation (degrees).
  --img-width IMG_WIDTH
                        Width of the sampled images.
  --img-height IMG_HEIGHT
                        Height of the sampled images.
  --workers WORKERS     Number of concurrent outgoing requests.
```

Note: Use [this online tool](https://www.acscdg.com/) to find distances and bearings of the ROI.
