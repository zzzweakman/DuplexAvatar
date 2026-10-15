"""Bounded browser delivery: controls first, ordered speech, latest idle clip."""
import asyncio
import time
from collections import deque


class MediaWindow:
    """Bound data handed to TCP until the browser confirms it has decoded it."""
    def __init__(self, limit=2, unit_ms=960, max_limit=6):
        self.limit, self.pending = limit, {}
        self.minimum, self.max_limit, self.unit_ms, self.rtt_ms = limit, max_limit, unit_ms, 0
        self.available = asyncio.Event()

    async def acquire(self, sequence):
        while len(self.pending) >= self.limit:
            self.available.clear()
            await asyncio.wait_for(self.available.wait(), 30)
        self.pending[sequence] = time.perf_counter()

    def acknowledge(self, sequence):
        began = self.pending.pop(sequence, None)
        self.available.set()
        if began is None:
            return None
        elapsed = (time.perf_counter() - began) * 1000
        self.rtt_ms = .8 * self.rtt_ms + .2 * elapsed if self.rtt_ms else elapsed
        self.limit = max(self.minimum, min(self.max_limit, int((self.rtt_ms + 100) / self.unit_ms) + 2))
        return elapsed


class InputQueue:
    """Preserve speech; under load collapse only consecutive near-digital silence."""
    def __init__(self, limit=30):
        self.limit = limit
        self.items = deque()
        self.ready = asyncio.Event()

    def qsize(self):
        return len(self.items)

    def put(self, item):
        dropped = []
        if len(self.items) >= 2 and item["quiet"] and self.items[-1]["quiet"]:
            dropped.append(self.items.pop())
        if len(self.items) >= self.limit:
            raise RuntimeError("录音缓冲已满，请暂停说话，等待输入处理完成后重试。")
        self.items.append(item)
        self.ready.set()
        return dropped

    async def get(self):
        while not self.items:
            self.ready.clear()
            await self.ready.wait()
        return self.items.popleft()


class Outbox:
    def __init__(self, media_limit=6, control_limit=64):
        self.control, self.media = deque(), deque()
        self.idle = None
        self.media_limit, self.control_limit = media_limit, control_limit
        self.ready = asyncio.Event()

    def put(self, payload):
        kind = payload["type"]
        if kind == "idle":
            self.idle = payload
        else:
            queue, limit = (self.media, self.media_limit) if kind in ("media", "response.end") else (self.control, self.control_limit)
            if len(queue) >= limit:
                raise RuntimeError("接收端处理过慢，队列已满。请结束会话后重试。")
            queue.append(payload)
        self.ready.set()

    def discard_before(self, epoch):
        for queue in (self.control, self.media):
            kept = [item for item in queue if item.get("epoch", epoch) >= epoch]
            queue.clear()
            queue.extend(kept)
        self.idle = None

    async def get(self):
        while True:
            if self.control:
                return self.control.popleft()
            if self.media:
                return self.media.popleft()
            if self.idle is not None:
                item, self.idle = self.idle, None
                return item
            self.ready.clear()
            await self.ready.wait()
