"""
TOPIC: Concurrency (Threading, Multiprocessing, asyncio)

USE CASES:
  - I/O-bound: threading or asyncio (network, file, database)
  - CPU-bound: multiprocessing (number crunching, data processing)
  - Mixed workloads: hybrid approaches

KEY POINTS:
  - GIL prevents true parallelism in threads for CPU work
  - Threads share memory, processes don't
  - asyncio: single-threaded, event-driven
  - concurrent.futures: high-level API for both
"""

import threading
import multiprocessing
import concurrent.futures
import time
import queue
from typing import Any

# =====================
# Threading (I/O-bound)
# =====================

def io_task(task_id: int, duration: float) -> str:
    """Simulate I/O operation."""
    print(f"Task {task_id} starting")
    time.sleep(duration)  # Simulates I/O wait
    print(f"Task {task_id} done")
    return f"Result from task {task_id}"


def demo_threading() -> None:
    print("=== Threading (I/O-bound) ===\n")

    # Without threading (sequential)
    start = time.perf_counter()
    results = []
    for i in range(3):
        results.append(io_task(i, 0.5))
    sequential_time = time.perf_counter() - start
    print(f"Sequential: {sequential_time:.2f}s\n")

    # With threading (concurrent)
    start = time.perf_counter()
    threads: list[threading.Thread] = []
    for i in range(3):
        t = threading.Thread(target=io_task, args=(i, 0.5))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()  # Wait for completion
    threaded_time = time.perf_counter() - start
    print(f"Threaded: {threaded_time:.2f}s")
    print(f"Speedup: {sequential_time/threaded_time:.1f}x\n")


# =====================
# Thread Synchronization
# =====================

class Counter:
    def __init__(self) -> None:
        self.value = 0
        self.lock = threading.Lock()

    def increment(self) -> None:
        with self.lock:  # Thread-safe
            current = self.value
            time.sleep(0.0001)  # Simulate work
            self.value = current + 1


def demo_thread_sync() -> None:
    print("=== Thread Synchronization ===\n")

    counter = Counter()
    threads = []

    for _ in range(100):
        t = threading.Thread(target=counter.increment)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Final value (with lock): {counter.value}")
    print("Without lock, race conditions would cause incorrect values\n")


# =====================
# Thread Pool
# =====================

def demo_thread_pool() -> None:
    print("=== Thread Pool ===\n")

    urls = [f"http://example.com/page{i}" for i in range(5)]

    def fetch_url(url: str) -> dict[str, str]:
        time.sleep(0.3)  # Simulate network
        return {"url": url, "status": "ok"}

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # map() for simple cases
        results = list(executor.map(fetch_url, urls))

    elapsed = time.perf_counter() - start
    print(f"Fetched {len(results)} URLs in {elapsed:.2f}s")
    print(f"(5 URLs, 3 workers, 0.3s each = ~0.6s expected)\n")


# =====================
# Multiprocessing (CPU-bound)
# =====================

def cpu_task(n: int) -> int:
    """CPU-intensive computation."""
    return sum(i * i for i in range(n))


def demo_multiprocessing() -> None:
    print("=== Multiprocessing (CPU-bound) ===\n")

    numbers = [5_000_000] * 4

    # Sequential
    start = time.perf_counter()
    results = [cpu_task(n) for n in numbers]
    sequential_time = time.perf_counter() - start
    print(f"Sequential: {sequential_time:.2f}s")

    # Parallel with ProcessPoolExecutor
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(cpu_task, numbers))
    parallel_time = time.perf_counter() - start
    print(f"Parallel: {parallel_time:.2f}s")
    print(f"Speedup: {sequential_time/parallel_time:.1f}x\n")


# =====================
# Future Objects
# =====================

def demo_futures() -> None:
    print("=== Future Objects ===\n")

    def compute(x: int) -> int:
        time.sleep(0.2)
        return x * x

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit returns Future immediately
        future1 = executor.submit(compute, 5)
        future2 = executor.submit(compute, 10)

        print("Futures submitted, doing other work...")

        # as_completed yields futures as they complete
        futures = [future1, future2]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(f"Got result: {result}")

        # Can also use wait()
        done, not_done = concurrent.futures.wait(
            [executor.submit(compute, i) for i in range(3)],
            return_when=concurrent.futures.FIRST_COMPLETED
        )
        print(f"First completed: {len(done)}, still running: {len(not_done)}")
    print()


# =====================
# Producer-Consumer with Queue
# =====================

def demo_producer_consumer() -> None:
    print("=== Producer-Consumer ===\n")

    task_queue: queue.Queue[int | None] = queue.Queue(maxsize=5)
    results: list[int] = []
    results_lock = threading.Lock()

    def producer(n: int) -> None:
        for i in range(n):
            task_queue.put(i)
            print(f"Produced: {i}")
            time.sleep(0.1)
        task_queue.put(None)  # Sentinel

    def consumer(name: str) -> None:
        while True:
            item = task_queue.get()
            if item is None:
                task_queue.put(None)  # Pass sentinel to next consumer
                break
            result = item * 2
            with results_lock:
                results.append(result)
            print(f"{name} consumed: {item} -> {result}")
            time.sleep(0.15)
            task_queue.task_done()

    # Start threads
    producer_thread = threading.Thread(target=producer, args=(5,))
    consumer_threads = [
        threading.Thread(target=consumer, args=(f"Consumer{i}",))
        for i in range(2)
    ]

    producer_thread.start()
    for t in consumer_threads:
        t.start()

    producer_thread.join()
    for t in consumer_threads:
        t.join()

    print(f"Results: {sorted(results)}\n")


# =====================
# Thread-Local Data
# =====================

def demo_thread_local() -> None:
    print("=== Thread-Local Data ===\n")

    local_data = threading.local()

    def worker(value: int) -> None:
        local_data.value = value
        time.sleep(0.1)
        print(f"Thread {threading.current_thread().name}: {local_data.value}")

    threads = [
        threading.Thread(target=worker, args=(i,), name=f"Worker-{i}")
        for i in range(3)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print()


# =====================
# When to Use What
# =====================

def print_summary() -> None:
    print("=== When to Use What ===\n")

    summary = """
╔════════════════════╦═══════════════════════════════════════════╗
║ Workload Type      ║ Best Approach                             ║
╠════════════════════╬═══════════════════════════════════════════╣
║ I/O-bound          ║ asyncio (best) or threading               ║
║ CPU-bound          ║ multiprocessing                           ║
║ Mixed              ║ ProcessPoolExecutor + asyncio/threading   ║
║ Simple parallelism ║ concurrent.futures (ThreadPool/ProcessPool)║
╚════════════════════╩═══════════════════════════════════════════╝

Key Points:
- GIL: Threads can't run Python code in parallel (only I/O)
- Processes: Full parallelism but higher overhead, no shared memory
- asyncio: Single-threaded, best for many concurrent I/O operations
- For web scraping: asyncio with aiohttp
- For data processing: multiprocessing or joblib
- For simple cases: concurrent.futures
"""
    print(summary)


# =====================
# Demo
# =====================

if __name__ == "__main__":
    print("=== Concurrency Demo ===\n")

    demo_threading()
    demo_thread_sync()
    demo_thread_pool()

    # Note: multiprocessing demo needs to be in main guard
    demo_multiprocessing()

    demo_futures()
    demo_producer_consumer()
    demo_thread_local()
    print_summary()
