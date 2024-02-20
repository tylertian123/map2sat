import asyncio
import aiohttp
import geopy.distance as distance
import pathlib

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
                     num_pts_h: int, num_pts_v: int, zoom: int = 16, bearing: int = 0, width: int = 256, height: int = 256) -> None:
    map_dir = pathlib.Path(map_dir)
    sat_dir = pathlib.Path(sat_dir)
    map_dir.mkdir(parents=True, exist_ok=True)
    sat_dir.mkdir(parents=True, exist_ok=True)

    poi = get_poi(roi[0], roi[1], roi[2], roi[3], num_pts_h, num_pts_v)
    async with aiohttp.ClientSession() as session:
        for i, (lat, lon) in enumerate(poi):
            print(f"Sampling coordinates ({lat:.6f}, {lon:.6f}) ({(i + 1) / len(poi) * 100:.2f}% complete)")
            try:
                map_img = await mapbox.request_image(session, "map", lat, lon, zoom, bearing, width, height)
                sat_img = await mapbox.request_image(session, "sat", lat, lon, zoom, bearing, width, height)
            except (aiohttp.ClientResponseError, ValueError) as e:
                print("\tError:", e)
                print("\tSkipping this point!")
                continue
            map_path = map_dir / f"map_{lat:.6f}_{lon:.6f}_zoom{zoom}.png"
            sat_path = sat_dir / f"sat_{lat:.6f}_{lon:.6f}_zoom{zoom}.png"
            map_img.save(map_path)
            sat_img.save(sat_path)
            print("\tSaved map as", map_path)
            print("\tSaved sat as", sat_path)

    print("Sampling completed.")


async def main():
    await sample_roi("data/map", "data/sat", ((43.700252, -79.50534), 12, 7.5, 24), 3, 3)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
