"""
TOPIC: asyncio & Async Programming

USE CASES:
  - I/O-bound operations (network, file, database)
  - Web servers, API clients
  - Concurrent tasks without threads
  - Rate-limited API calls

KEY POINTS:
  - async def creates coroutine function
  - await pauses until result ready
  - asyncio.gather() runs multiple coroutines concurrently
  - NOT for CPU-bound work (use multiprocessing)
  - One event loop per thread
"""

import asyncio
import time
from typing import Any
from dataclasses import dataclass

# =====================
# Basic Coroutines
# =====================

async def say_hello(name: str, delay: float = 1.0) -> str:
    """Simple coroutine that waits and returns."""
    await asyncio.sleep(delay)  # Non-blocking sleep
    return f"Hello, {name}!"


async def fetch_data(url: str) -> dict[str, Any]:
    """Simulate fetching data from URL."""
    print(f"Fetching {url}...")
    await asyncio.sleep(0.5)  # Simulate network delay
    return {"url": url, "data": "response_data"}


# =====================
# Running Coroutines
# =====================

async def demo_basic() -> None:
    print("=== Basic Coroutines ===")

    # Single coroutine
    result = await say_hello("World")
    print(result)

    # Sequential (slow)
    start = time.perf_counter()
    await say_hello("Alice", 0.5)
    await say_hello("Bob", 0.5)
    print(f"Sequential: {time.perf_counter() - start:.2f}s")

    # Concurrent with gather (fast)
    start = time.perf_counter()
    results = await asyncio.gather(
        say_hello("Alice", 0.5),
        say_hello("Bob", 0.5),
        say_hello("Charlie", 0.5),
    )
    print(f"Concurrent: {time.perf_counter() - start:.2f}s")
    print(f"Results: {results}")
    print()


# =====================
# Tasks
# =====================

async def demo_tasks() -> None:
    print("=== Tasks ===")

    # Create task (starts running immediately)
    task1 = asyncio.create_task(say_hello("Task1", 0.3))
    task2 = asyncio.create_task(say_hello("Task2", 0.2))

    print("Tasks created, doing other work...")
    await asyncio.sleep(0.1)
    print("Other work done")

    # Wait for tasks
    result1 = await task1
    result2 = await task2
    print(f"Results: {result1}, {result2}")

    # Cancel a task
    task3 = asyncio.create_task(say_hello("Task3", 10))
    await asyncio.sleep(0.1)
    task3.cancel()
    try:
        await task3
    except asyncio.CancelledError:
        print("Task3 was cancelled")
    print()


# =====================
# Timeouts
# =====================

async def slow_operation() -> str:
    await asyncio.sleep(5)
    return "Done"


async def demo_timeouts() -> None:
    print("=== Timeouts ===")

    # wait_for with timeout
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=0.5)
        print(f"Result: {result}")
    except asyncio.TimeoutError:
        print("Operation timed out!")

    # timeout context manager (Python 3.11+)
    try:
        async with asyncio.timeout(0.5):
            await slow_operation()
    except asyncio.TimeoutError:
        print("Context timeout!")
    print()


# =====================
# Semaphore (Rate Limiting)
# =====================

async def fetch_with_limit(
    url: str,
    semaphore: asyncio.Semaphore
) -> dict[str, Any]:
    async with semaphore:
        print(f"Fetching {url}")
        await asyncio.sleep(0.5)
        return {"url": url}


async def demo_semaphore() -> None:
    print("=== Semaphore (Rate Limiting) ===")

    urls = [f"http://api.example.com/item/{i}" for i in range(6)]

    # Limit to 2 concurrent requests
    semaphore = asyncio.Semaphore(2)

    start = time.perf_counter()
    results = await asyncio.gather(*[
        fetch_with_limit(url, semaphore) for url in urls
    ])
    elapsed = time.perf_counter() - start

    print(f"Fetched {len(results)} items in {elapsed:.2f}s")
    print(f"(6 items, 2 concurrent, 0.5s each = ~1.5s expected)")
    print()


# =====================
# Queue for Producer/Consumer
# =====================

async def producer(queue: asyncio.Queue[int], n: int) -> None:
    for i in range(n):
        await queue.put(i)
        print(f"Produced: {i}")
        await asyncio.sleep(0.1)
    await queue.put(-1)  # Sentinel to stop consumer


async def consumer(queue: asyncio.Queue[int], name: str) -> None:
    while True:
        item = await queue.get()
        if item == -1:
            queue.task_done()
            break
        print(f"{name} consumed: {item}")
        await asyncio.sleep(0.2)
        queue.task_done()


async def demo_queue() -> None:
    print("=== Async Queue ===")

    queue: asyncio.Queue[int] = asyncio.Queue(maxsize=3)

    # Run producer and consumer concurrently
    await asyncio.gather(
        producer(queue, 5),
        consumer(queue, "Consumer"),
    )
    print()


# =====================
# Lock
# =====================

@dataclass
class SharedState:
    value: int = 0


async def increment(state: SharedState, lock: asyncio.Lock) -> None:
    async with lock:
        current = state.value
        await asyncio.sleep(0.01)  # Simulate processing
        state.value = current + 1


async def demo_lock() -> None:
    print("=== Async Lock ===")

    state = SharedState()
    lock = asyncio.Lock()

    # Without lock, race condition would occur
    await asyncio.gather(*[increment(state, lock) for _ in range(10)])
    print(f"Final value (with lock): {state.value}")
    print()


# =====================
# Event (Signaling)
# =====================

async def waiter(event: asyncio.Event, name: str) -> None:
    print(f"{name} waiting for event...")
    await event.wait()
    print(f"{name} got the event!")


async def demo_event() -> None:
    print("=== Async Event ===")

    event = asyncio.Event()

    # Start waiters
    task1 = asyncio.create_task(waiter(event, "Waiter1"))
    task2 = asyncio.create_task(waiter(event, "Waiter2"))

    await asyncio.sleep(0.5)
    print("Setting event!")
    event.set()  # All waiters proceed

    await asyncio.gather(task1, task2)
    print()


# =====================
# Exception Handling
# =====================

async def might_fail(should_fail: bool) -> str:
    await asyncio.sleep(0.1)
    if should_fail:
        raise ValueError("Operation failed!")
    return "Success"


async def demo_exceptions() -> None:
    print("=== Exception Handling ===")

    # gather with return_exceptions
    results = await asyncio.gather(
        might_fail(False),
        might_fail(True),
        might_fail(False),
        return_exceptions=True  # Don't raise, return exceptions
    )

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i} failed: {result}")
        else:
            print(f"Task {i} succeeded: {result}")
    print()


# =====================
# TaskGroup (Python 3.11+)
# =====================

async def demo_taskgroup() -> None:
    print("=== TaskGroup (Python 3.11+) ===")

    results: list[str] = []

    async with asyncio.TaskGroup() as tg:
        # All tasks created in group are awaited at end
        task1 = tg.create_task(say_hello("A", 0.2))
        task2 = tg.create_task(say_hello("B", 0.3))
        task3 = tg.create_task(say_hello("C", 0.1))

    # All tasks completed here
    results = [task1.result(), task2.result(), task3.result()]
    print(f"Results: {results}")
    print()


# =====================
# async for / async with
# =====================

class AsyncRange:
    """Async iterator example."""

    def __init__(self, start: int, stop: int) -> None:
        self.start = start
        self.stop = stop

    def __aiter__(self) -> "AsyncRange":
        self.current = self.start
        return self

    async def __anext__(self) -> int:
        if self.current >= self.stop:
            raise StopAsyncIteration
        await asyncio.sleep(0.1)  # Simulate async operation
        result = self.current
        self.current += 1
        return result


async def demo_async_iteration() -> None:
    print("=== async for ===")

    async for i in AsyncRange(0, 5):
        print(f"Got: {i}")
    print()


# =====================
# Run Sync Code in Thread
# =====================

def blocking_io_operation(data: str) -> str:
    """Simulate blocking I/O."""
    time.sleep(0.5)  # Blocking!
    return f"Processed: {data}"


async def demo_run_in_executor() -> None:
    print("=== Run Blocking Code ===")

    loop = asyncio.get_running_loop()

    # Run blocking function in thread pool
    start = time.perf_counter()
    results = await asyncio.gather(
        loop.run_in_executor(None, blocking_io_operation, "A"),
        loop.run_in_executor(None, blocking_io_operation, "B"),
        loop.run_in_executor(None, blocking_io_operation, "C"),
    )
    elapsed = time.perf_counter() - start

    print(f"Results: {results}")
    print(f"Concurrent blocking calls: {elapsed:.2f}s")
    print()


# =====================
# Main
# =====================

async def main() -> None:
    print("=== asyncio Demo ===\n")

    await demo_basic()
    await demo_tasks()
    await demo_timeouts()
    await demo_semaphore()
    await demo_queue()
    await demo_lock()
    await demo_event()
    await demo_exceptions()
    await demo_taskgroup()
    await demo_async_iteration()
    await demo_run_in_executor()


if __name__ == "__main__":
    asyncio.run(main())
