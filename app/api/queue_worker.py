from __future__ import annotations

import os

from redis import Redis
from rq import Connection, Queue, Worker

from app.api.db import init_db


def main() -> None:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    init_db()
    conn = Redis.from_url(redis_url)
    with Connection(conn):
        queue = Queue("basketball-jobs")
        worker = Worker([queue])
        worker.work()


if __name__ == "__main__":
    main()
