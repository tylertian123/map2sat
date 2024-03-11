import argparse
import asyncio
import pathlib
import random

import aiohttp
import geopy.distance as distance

import mapbox


def get_poi(roi_coords: tuple[float, float], width: float, height: float, rotation: float, num_pts_h: int, num_pts_v: int) -> list[tuple[float, float]]:
    """
    Get coordinates of points of interest within a region of interest.

    Returns a list of (lat, lon) coordinates of each POI (total num_pts_h * num_pts_v points).
    One point will always be at the ROI coordinates.

    roi_coords: Region of interest (lat, lon), top-left corner of the ROI.
    width: Width of the ROI in km.
    height: Height of the ROI in km.
    rotation: How much the ROI rectangle is rotated (from being aligned with lat/lon lines) in degrees.
              Positive rotation is counterclockwise.
    num_pts_h: Number of points to collect horizontally.
    num_pts_v: Number of points to collect vertically.
    """
    dx = width / (num_pts_h - 1) if num_pts_h > 1 else 0
    dy = height / (num_pts_v - 1) if num_pts_v > 1 else 0
    poi = []
    for i in range(num_pts_v):
        # Calculate starting location for this row
        # Note bearing is measured clockwise from North
        start = distance.distance(kilometers=dy * i).destination(roi_coords, bearing=180 - rotation)
        for j in range(num_pts_h):
            poi.append(distance.distance(kilometers=dx * j).destination(start, bearing=90 - rotation))
    return [(p[0], p[1]) for p in poi]


async def sample_roi(map_dir: str, sat_dir: float, roi: tuple[tuple[float, float], float, float, float],
                     num_pts_h: int, num_pts_v: int, zoom: int = 16, bearing: int = 0, width: int = 256, height: int = 256,
                     randomize_bearing: bool = False, num_workers: int = 1) -> None:
    map_dir = pathlib.Path(map_dir)
    sat_dir = pathlib.Path(sat_dir)
    map_dir.mkdir(parents=True, exist_ok=True)
    sat_dir.mkdir(parents=True, exist_ok=True)

    if num_workers > 1:
        print("INFO: Using more than 1 worker")

    poi = get_poi(roi[0], roi[1], roi[2], roi[3], num_pts_h, num_pts_v)
    # Split into subsets for each worker
    poi_subsets = []
    pts_per_worker = len(poi) // num_workers
    for i in range(num_workers):
        if i < num_workers - 1:
            poi_subsets.append(poi[i * pts_per_worker:(i + 1) * pts_per_worker])
        else:
            poi_subsets.append(poi[i * pts_per_worker:])
    
    total_sampled = 0

    async def sample_subset(session, subset: list[tuple[float, float]]):
        nonlocal total_sampled, bearing
        for lat, lon in subset:
            total_sampled += 1
            try:
                if randomize_bearing:
                    bearing = random.randint(0, 359)
                map_img = await mapbox.request_image(session, "map", lat, lon, zoom, bearing, width, height)
                sat_img = await mapbox.request_image(session, "sat", lat, lon, zoom, bearing, width, height)
            except (aiohttp.ClientResponseError, ValueError) as e:
                print("\tError:", e)
                print(f"\tSkipping coordinates ({lat:.6f}, {lon:.6f})!")
                continue
            print(f"Sampled coordinates ({lat:.6f}, {lon:.6f}) ({total_sampled / len(poi) * 100:.2f}% complete)")
            map_path = map_dir / f"map_{lat:.6f}_{lon:.6f}_zoom{zoom}.png"
            sat_path = sat_dir / f"sat_{lat:.6f}_{lon:.6f}_zoom{zoom}.png"
            map_img.save(map_path)
            sat_img.save(sat_path)
            print("\tSaved map as", map_path)
            print("\tSaved sat as", sat_path)
    
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(sample_subset(session, subset) for subset in poi_subsets))

    print("Sampling completed.")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lat", type=float, help="Latitude of the top-left corner of the ROI.")
    parser.add_argument("lon", type=float, help="Longitude of the top-left corner of the ROI.")
    parser.add_argument("width", type=float, help="Width of the ROI in km.")
    parser.add_argument("height", type=float, help="Height of the ROI in km.")
    parser.add_argument("h_samples", type=int, help="Sample grid size (horizontal).")
    parser.add_argument("v_samples", type=int, help="Sample grid size (vertical).")
    
    parser.add_argument("--rotation", default=0, type=float, help="How much the ROI rectangle is rotated (from being aligned with lat/lon lines) in degrees. Positive rotation is counterclockwise.")

    parser.add_argument("--map-dir", default="data/map", type=str, help="Directory to store the map images.")
    parser.add_argument("--sat-dir", default="data/sat", type=str, help="Directory to store the satellite images.")
    parser.add_argument("--zoom", default=16, type=int, help="Zoom level.")
    parser.add_argument("--bearing", default=0, type=int, help="Map rotation (degrees).")
    parser.add_argument("--img-width", default=256, type=int, help="Width of the sampled images.")
    parser.add_argument("--img-height", default=256, type=int, help="Height of the sampled images.")
    parser.add_argument("--randomize-bearing", action="store_true", help="Specify this flag to randomize the bearing of each image taken.")

    parser.add_argument("--workers", default=1, type=int, help="Number of concurrent outgoing requests.")

    args = parser.parse_args()

    await sample_roi(args.map_dir, args.sat_dir, ((args.lat, args.lon), args.width, args.height, args.rotation), args.h_samples, args.v_samples, args.zoom, args.bearing, args.img_width, args.img_height, args.randomize_bearing, args.workers)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
