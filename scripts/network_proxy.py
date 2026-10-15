"""Local test proxy with shared bandwidth and pipelined propagation delay.

Only WebSockets are shaped; bootstrap HTTP is passed through. Both application
sockets share each direction's byte budget, as with one forwarded connection.
"""
import argparse
import asyncio
from contextlib import suppress
from pathlib import Path
import ssl
import time

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
import httpx
import uvicorn
import websockets

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8444)
parser.add_argument("--rtt-ms", type=int, default=2400)
parser.add_argument("--down-mbps", type=float, default=4)
parser.add_argument("--up-mbps", type=float, default=1)
args = parser.parse_args()
app = FastAPI()


class Link:
    def __init__(self, mbps):
        self.rate, self.available = mbps * 1e6 / 8, 0

    def delivery_time(self, data):
        size = len(data.encode()) if isinstance(data, str) else len(data)
        self.available = max(time.monotonic(), self.available) + size / self.rate
        return self.available + args.rtt_ms / 2000


up, down = Link(args.up_mbps), Link(args.down_mbps)


async def relay(receive, send, link):
    pending = asyncio.Queue(maxsize=32)
    async def read():
        while True:
            data = await receive()
            await pending.put((link.delivery_time(data), data))
    async def write():
        while True:
            due, data = await pending.get()
            await asyncio.sleep(max(0, due - time.monotonic()))
            await send(data)
    tasks = [asyncio.create_task(read()), asyncio.create_task(write())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@app.websocket("/{path:path}")
async def socket(client: WebSocket, path: str):
    await client.accept()
    tasks = []
    try:
        async with websockets.connect(f"wss://127.0.0.1:8443/{path}", ssl=ssl._create_unverified_context(),
                                      compression=None, max_size=32 * 1024 * 1024, ping_timeout=120) as backend:
            tasks = [asyncio.create_task(relay(client.receive_text, backend.send, up)),
                     asyncio.create_task(relay(backend.recv, client.send_text, down))]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        with suppress(Exception):
            await client.close()


@app.get("/{path:path}")
async def http(request: Request, path: str):
    async with httpx.AsyncClient(verify=False, trust_env=False) as client:
        result = await client.get(f"https://127.0.0.1:8443/{path}", params=request.query_params)
    return Response(result.content, status_code=result.status_code,
                    media_type=result.headers.get("content-type"))


if __name__ == "__main__":
    certs = ROOT.parent / "MiniCPM-o-Demo/local_deploy/certs"
    uvicorn.run(app, host="127.0.0.1", port=args.port, ssl_certfile=str(certs / "cert.pem"),
                ssl_keyfile=str(certs / "key.pem"), ws_per_message_deflate=False, ws_max_size=16 * 1024 * 1024,
                log_level="warning")
