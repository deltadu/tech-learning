"""
TOPIC: Generators & Iterators

USE CASES:
  - Processing large files line by line
  - Infinite sequences
  - Lazy evaluation (compute on demand)
  - Memory-efficient data pipelines

KEY POINTS:
  - yield pauses function and returns value
  - Generator = iterator created by function with yield
  - Generator expressions: (x for x in items)
  - send() to pass values into generator
  - yield from delegates to sub-generator
"""

from typing import Iterator, Generator, Iterable, TypeVar
import itertools

T = TypeVar("T")

# =====================
# Basic Generator
# =====================


def countdown(n: int) -> Generator[int, None, None]:
    """Generator that counts down from n to 1."""
    print(f"Starting countdown from {n}")
    while n > 0:
        yield n  # Pause here, return n
        n -= 1
    print("Blastoff!")


def fibonacci() -> Generator[int, None, None]:
    """Infinite fibonacci sequence."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


# =====================
# Generator Expression
# =====================


def demo_generator_expressions() -> None:
    print("=== Generator Expressions ===")

    # List comprehension (eager, uses memory)
    squares_list = [x**2 for x in range(10)]

    # Generator expression (lazy, memory efficient)
    squares_gen = (x**2 for x in range(10))

    print(f"List: {squares_list}")
    print(f"Generator: {squares_gen}")  # Shows generator object
    print(f"Consumed: {list(squares_gen)}")

    # Memory comparison
    # list(range(1_000_000_000))  # Would use ~8GB RAM!
    # (x for x in range(1_000_000_000))  # Uses almost no RAM
    print()


# =====================
# Reading Large Files
# =====================


def read_large_file(filepath: str) -> Generator[str, None, None]:
    """Memory-efficient file reading."""
    with open(filepath) as f:
        for line in f:
            yield line.strip()


def grep(pattern: str, lines: Iterable[str]) -> Generator[str, None, None]:
    """Filter lines containing pattern."""
    for line in lines:
        if pattern in line:
            yield line


# Pipeline: read_large_file("log.txt") | grep("ERROR") | process


# =====================
# Generator Pipeline
# =====================


def integers(start: int = 0) -> Generator[int, None, None]:
    """Infinite integers."""
    n = start
    while True:
        yield n
        n += 1


def take(n: int, iterable: Iterable[T]) -> Generator[T, None, None]:
    """Take first n items."""
    for i, item in enumerate(iterable):
        if i >= n:
            return
        yield item


def filter_gen(predicate, iterable: Iterable[T]) -> Generator[T, None, None]:
    """Filter items by predicate."""
    for item in iterable:
        if predicate(item):
            yield item


def map_gen(func, iterable: Iterable[T]) -> Generator:
    """Map function over items."""
    for item in iterable:
        yield func(item)


def demo_pipeline() -> None:
    print("=== Generator Pipeline ===")

    # Compose generators: lazy evaluation
    pipeline = take(
        5,
        filter_gen(
            lambda x: x % 2 == 0, map_gen(lambda x: x**2, integers(1))
        ),
    )

    print(f"First 5 even squares: {list(pipeline)}")
    print()


# =====================
# send() - Coroutine Pattern
# =====================


def accumulator() -> Generator[int, int, None]:
    """Generator that receives values and accumulates them."""
    total = 0
    while True:
        value = yield total  # yield current, receive next
        if value is None:
            break
        total += value


def running_average() -> Generator[float | None, float, None]:
    """Compute running average."""
    total = 0.0
    count = 0
    average: float | None = None
    while True:
        value = yield average
        if value is None:
            break
        total += value
        count += 1
        average = total / count


def demo_send() -> None:
    print("=== Generator send() ===")

    acc = accumulator()
    next(acc)  # Prime the generator (run to first yield)

    print(f"Send 10: {acc.send(10)}")
    print(f"Send 20: {acc.send(20)}")
    print(f"Send 30: {acc.send(30)}")

    print()

    avg = running_average()
    next(avg)  # Prime

    for value in [10, 20, 30, 40, 50]:
        result = avg.send(value)
        print(f"After {value}: average = {result}")
    print()


# =====================
# yield from (Delegation)
# =====================


def flatten(nested: list) -> Generator:
    """Recursively flatten nested lists."""
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)  # Delegate to sub-generator
        else:
            yield item


def chain(*iterables: Iterable[T]) -> Generator[T, None, None]:
    """Chain multiple iterables (like itertools.chain)."""
    for iterable in iterables:
        yield from iterable


def demo_yield_from() -> None:
    print("=== yield from ===")

    nested = [1, [2, 3, [4, 5]], 6, [7, 8]]
    print(f"Flatten {nested}: {list(flatten(nested))}")

    print(f"chain([1,2], [3,4], [5]): {list(chain([1, 2], [3, 4], [5]))}")
    print()


# =====================
# Generator Cleanup
# =====================


def managed_generator() -> Generator[int, None, None]:
    """Generator with cleanup."""
    print("Generator started")
    try:
        for i in range(10):
            yield i
    except GeneratorExit:
        print("Generator closed early")
    finally:
        print("Generator cleanup")


def demo_cleanup() -> None:
    print("=== Generator Cleanup ===")

    gen = managed_generator()
    print(f"Got: {next(gen)}")
    print(f"Got: {next(gen)}")
    gen.close()  # Triggers GeneratorExit
    print()


# =====================
# throw() - Inject Exception
# =====================


def interruptible() -> Generator[int, None, None]:
    """Generator that handles injected exceptions."""
    n = 0
    while True:
        try:
            yield n
            n += 1
        except ValueError as e:
            print(f"Caught: {e}")
            n = 0  # Reset on error


def demo_throw() -> None:
    print("=== Generator throw() ===")

    gen = interruptible()
    print(f"Got: {next(gen)}")
    print(f"Got: {next(gen)}")
    print(f"Got: {next(gen)}")
    print(f"After throw: {gen.throw(ValueError, 'Reset!')}")
    print(f"Got: {next(gen)}")
    print()


# =====================
# Iterator Protocol
# =====================


class Range:
    """Custom range-like iterator."""

    def __init__(self, start: int, stop: int) -> None:
        self.start = start
        self.stop = stop

    def __iter__(self) -> Iterator[int]:
        current = self.start
        while current < self.stop:
            yield current
            current += 1


class InfiniteCounter:
    """Iterator class (not generator)."""

    def __init__(self, start: int = 0) -> None:
        self.current = start

    def __iter__(self) -> "InfiniteCounter":
        return self

    def __next__(self) -> int:
        result = self.current
        self.current += 1
        return result


def demo_iterator_protocol() -> None:
    print("=== Iterator Protocol ===")

    print(f"Custom Range(3, 7): {list(Range(3, 7))}")

    counter = InfiniteCounter(100)
    print(
        first_5 = [next(counter) for _ in range(5)]
    print(f"First 5 from InfiniteCounter(100): {first_5}"
    )
    print()


# =====================
# Practical: Batching
# =====================


def batch(
    iterable: Iterable[T], size: int
) -> Generator[list[T], None, None]:
    """Yield batches of given size."""
    batch_items: list[T] = []
    for item in iterable:
        batch_items.append(item)
        if len(batch_items) == size:
            yield batch_items
            batch_items = []
    if batch_items:  # Yield remaining items
        yield batch_items


def demo_batching() -> None:
    print("=== Batching ===")

    items = range(10)
    for b in batch(items, 3):
        print(f"Batch: {b}")
    print()


# =====================
# Demo
# =====================

if __name__ == "__main__":
    print("=== Generators Demo ===\n")

    # Basic countdown
    print("--- Countdown ---")
    for n in countdown(3):
        print(n)
    print()

    # Fibonacci
    print("--- Fibonacci ---")
    fib = fibonacci()
    print(f"First 10 fibonacci: {[next(fib) for _ in range(10)]}")
    print()

    demo_generator_expressions()
    demo_pipeline()
    demo_send()
    demo_yield_from()
    demo_cleanup()
    demo_throw()
    demo_iterator_protocol()
    demo_batching()
