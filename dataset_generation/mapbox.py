"""
This module requests images of maps and satellite views from Mapbox.

Note asyncio is used here so requests can potentially be made in parallel.
"""

import io
import pathlib

import aiohttp
import PIL.Image

# Style URLs for the map and satellite styles
# Split into username and ID
STYLE_MAP = ("tylertian", "clsh7zutm03i701qqd2q3he5f")
STYLE_SAT = ("mapbox", "satellite-v9")

# Height of mapbox watermark for cropping
WATERMARK_HEIGHT = 20

# Mapbox API key
# Automatically read from mapbox_api_key.txt stored in the same dir as the script
API_KEY = None
with open(pathlib.Path(__file__).parent / "mapbox_api_key.txt", "r") as f:
    API_KEY = f.read().strip()

def format_url(map_type: str, lat: float, lon: float, zoom: float, bearing: int, width: int, height: int, token: str = API_KEY) -> str:
    """
    Format a URL for mapbox's static images API.

    Returns formatted URL.

    map_type: Style of map, either "map" or "satellite"
    lat: Latitude
    lon: Longitude
    zoom: Zoom level
    bearing: Map rotation (degrees)
    width: Width in pixels
    height: Height in pixels
    token: Mapbox API token
    """
    username, style_id = (STYLE_MAP if map_type == "map" else STYLE_SAT)
    return f"https://api.mapbox.com/styles/v1/{username}/{style_id}/static/{lon},{lat},{zoom:g},{bearing}/{width}x{height}?access_token={token}"

async def request_image(session: aiohttp.ClientSession, map_type: str, lat: float, lon: float, zoom: float, bearing: int, width: int, height: int, crop_watermark: bool = True, token: str = API_KEY) -> PIL.Image.Image:
    """
    Send a request and download the image from mapbox.

    Returns a PIL image.

    Raises aiohttp.ClientResponseError on request error, or ValueError if content type is unrecognized.

    session: aiohttp client session
    crop_watermark: If true, will request a larger image and crop out the mapbox watermark.
    For other input args see format_url().
    """
    if crop_watermark:
        height += WATERMARK_HEIGHT
    async with session.get(format_url(map_type, lat, lon, zoom, bearing, width, height, token)) as resp:
        resp.raise_for_status()
        if resp.content_type == "image/png":
            ext = "png"
        elif resp.content_type == "image/jpeg":
            ext = "jpeg"
        else:
            raise ValueError("Unrecognized MIME type: " + resp.content_type)
        raw = await resp.read()
        img = PIL.Image.open(io.BytesIO(raw), formats=[ext])
        if crop_watermark:
            img = img.crop((0, 0, width, height - WATERMARK_HEIGHT))
        return img
