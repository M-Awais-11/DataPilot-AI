"""
SSE (Server-Sent Events) service.
Broadcasts progress events to subscribed HTTP clients.
Supports both Redis pub/sub and in-memory queue broker for direct local execution without Redis/Docker.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator
from collections import defaultdict

logger = logging.getLogger(__name__)

# In-memory broker queues: experiment_id -> list of asyncio.Queue
_memory_subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
_event_history: dict[str, list[dict]] = defaultdict(list)


def publish_event_sync(
    experiment_id: str,
    message: str,
    agent: str = "system",
    level: str = "info",
) -> None:
    """Publish a progress event (called synchronously from pipeline runner)."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "message": message,
        "level": level,
    }
    # Store in recent history
    _event_history[experiment_id].append(event)
    
    # Broadcast to in-memory queues
    subscribers = _memory_subscribers.get(experiment_id, [])
    for q in list(subscribers):
        try:
            q.put_nowait(event)
        except Exception:
            pass

    # Optional Redis fallback
    try:
        from app.config.settings import settings
        if hasattr(settings, "redis_url") and settings.redis_url:
            import redis
            r = redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=0.5)
            r.publish(f"experiment_progress:{experiment_id}", json.dumps(event))
            r.close()
    except Exception:
        pass


async def publish_event(
    experiment_id: str,
    message: str,
    agent: str = "system",
    level: str = "info",
) -> None:
    publish_event_sync(experiment_id, message, agent, level)


async def event_stream(experiment_id: str) -> AsyncIterator[str]:
    """Yields SSE-formatted strings for a given experiment."""
    q: asyncio.Queue = asyncio.Queue()
    _memory_subscribers[experiment_id].append(q)

    # Send initial connection
    yield f"data: {json.dumps({'type': 'connected', 'experiment_id': experiment_id})}\n\n"

    # Replay past events if any
    for past in _event_history.get(experiment_id, []):
        yield f"data: {json.dumps(past)}\n\n"

    try:
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("level") in ("done", "fatal"):
                    break
            except asyncio.TimeoutError:
                # Keep-alive heartbeat comment
                yield ": keep-alive\n\n"
    except asyncio.CancelledError:
        pass
    finally:
        if q in _memory_subscribers.get(experiment_id, []):
            _memory_subscribers[experiment_id].remove(q)
