import asyncio
import websockets
import json

async def test():
    uri = "ws://localhost:8000/scan"

    async with websockets.connect(uri) as ws:
        image_path = "D:\\OneDrive\\Desktop\\GUC\\Courses\\Semster 10\\DLCV\\Project\\X-CV\\Model Fine-tuning\\Models Workspaces\\X-Ray Region Classification\\input\\Datasets - Unsplit\\X-Ray Region Classification\\Humerus\\19092.png"  # Path to your test image
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        await ws.send(image_bytes)  # send as bytes, not JSON

        while True:
            message = await ws.recv()
            data = json.loads(message)
            print(data)

            if data.get("status") in ("complete", "failed"):
                break

asyncio.run(test())