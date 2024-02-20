import asyncio
import aiohttp

import mapbox


async def main():

    async with aiohttp.ClientSession() as session:
        map_img = await mapbox.request_image(session, "map", 43.6644, -79.3927, 16, 0, 256, 256)
        sat_img = await mapbox.request_image(session, "sat", 43.6644, -79.3927, 16, 0, 256, 256)
    map_img.save("map.png")
    sat_img.save("sat.png")

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
