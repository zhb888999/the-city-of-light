import aiohttp
import asyncio
import time

import base64
import json

with open("test_img/4.png", 'rb') as f:
# with open("test_img/3_.jpg", 'rb') as f:
    image = f.read()


async def request():
    data = {
        "image": base64.b64encode(image).decode(),
        "box_threshold": 0.6
    }
    async with aiohttp.ClientSession() as session:
        async with session.post('http://127.0.0.1:8001/model/ocr', json=data) as resp:
            print(await resp.text())
        async with session.get('http://127.0.0.1:8001/config') as resp:
            print(await resp.text())

async def main():
    count = 1
    start = time.time()
    tasks = [asyncio.create_task(request()) for i in range(count)]
    [await task for task in tasks]
    end = time.time()
    avg = (end - start) / count
    tps = 1 / avg
    print(avg, tps)


if __name__ == '__main__':
    asyncio.run(main())