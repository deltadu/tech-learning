"""
TOPIC: Profiling & Optimization

USE CASES:
  - Finding performance bottlenecks
  - Memory profiling
  - Optimizing hot paths
  - Benchmarking changes

KEY POINTS:
  - Profile first, optimize later
  - cProfile for CPU profiling
  - memory_profiler for memory
  - timeit for micro-benchmarks
  - 80/20 rule: 80% time in 20% of code
"""

import cProfile
import pstats
import timeit
import time
import functools
import io
from typing import Callable, TypeVar, ParamSpec
import sys

P = ParamSpec("P")
R = TypeVar("R")

# =====================
# Simple Timing Decorator
# =====================


def timed(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to time function execution."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__}: {elapsed:.6f}s")
        return result

    return wrapper


# =====================
# Using timeit
# =====================


def demo_timeit() -> None:
    print("=== timeit ===")

    # Time a simple expression
    time_list = timeit.timeit("[x**2 for x in range(100)]", number=10000)
    time_gen = timeit.timeit("list(x**2 for x in range(100))", number=10000)

    print(f"List comprehension: {time_list:.4f}s")
    print(f"Generator + list(): {time_gen:.4f}s")

    # Time with setup
    setup = "data = list(range(1000))"
    stmt1 = "x in data"
    stmt2 = "data_set = set(data); x in data_set"

    t1 = timeit.timeit(stmt1, setup + "; x = 999", number=10000)
    t2 = timeit.timeit(stmt2, setup + "; x = 999", number=10000)

    print(f"\nList lookup: {t1:.6f}s")
    print(f"Set lookup (including conversion): {t2:.6f}s")
    print()


# =====================
# cProfile
# =====================


def slow_function() -> int:
    """Function with performance issues."""
    total = 0
    for i in range(1000):
        total += sum(range(i))  # Inefficient!
    return total


def fast_function() -> int:
    """Optimized version."""
    # sum(range(n)) = n*(n-1)/2
    return sum(i * (i - 1) // 2 for i in range(1000))


def demo_cprofile() -> None:
    print("=== cProfile ===")

    # Profile slow function
    profiler = cProfile.Profile()
    profiler.enable()
    slow_function()
    profiler.disable()

    # Print stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(10)
    print(stream.getvalue()[:1000])  # Truncate for demo

    # Compare times
    print("\n--- Timing Comparison ---")
    t1 = timeit.timeit(slow_function, number=10)
    t2 = timeit.timeit(fast_function, number=10)
    print(f"Slow: {t1:.4f}s")
    print(f"Fast: {t2:.4f}s")
    print(f"Speedup: {t1/t2:.1f}x")
    print()


# =====================
# Common Optimizations
# =====================


def demo_optimizations() -> None:
    print("=== Common Optimizations ===\n")

    n = 100000

    # 1. List vs Generator for intermediate results
    print("--- Intermediate Results ---")

    def with_list() -> int:
        squares = [x**2 for x in range(n)]
        return sum(squares)

    def with_generator() -> int:
        return sum(x**2 for x in range(n))

    t1 = timeit.timeit(with_list, number=100)
    t2 = timeit.timeit(with_generator, number=100)
    print(f"List: {t1:.4f}s, Generator: {t2:.4f}s")
    print()

    # 2. String concatenation
    print("--- String Concatenation ---")

    def concat_plus() -> str:
        s = ""
        for i in range(1000):
            s += str(i)
        return s

    def concat_join() -> str:
        return "".join(str(i) for i in range(1000))

    t1 = timeit.timeit(concat_plus, number=1000)
    t2 = timeit.timeit(concat_join, number=1000)
    print(f"Plus: {t1:.4f}s, Join: {t2:.4f}s")
    print()

    # 3. Lookup optimization
    print("--- Lookup Optimization ---")

    data = list(range(10000))
    data_set = set(data)

    def list_lookup() -> bool:
        return 9999 in data

    def set_lookup() -> bool:
        return 9999 in data_set

    t1 = timeit.timeit(list_lookup, number=10000)
    t2 = timeit.timeit(set_lookup, number=10000)
    print(f"List O(n): {t1:.4f}s, Set O(1): {t2:.4f}s")
    print()

    # 4. Local variable access
    print("--- Local vs Global ---")

    global_list = list(range(100))

    def use_global() -> int:
        total = 0
        for x in global_list:
            total += x
        return total

    def use_local() -> int:
        local_list = global_list  # Cache in local
        total = 0
        for x in local_list:
            total += x
        return total

    t1 = timeit.timeit(use_global, number=10000)
    t2 = timeit.timeit(use_local, number=10000)
    print(f"Global: {t1:.4f}s, Local: {t2:.4f}s")
    print()

    # 5. Avoid function calls in loops
    print("--- Function Call Overhead ---")

    def with_len_call() -> int:
        items = list(range(1000))
        total = 0
        for i in range(len(items)):
            total += items[i]
        return total

    def without_len_call() -> int:
        items = list(range(1000))
        n = len(items)
        total = 0
        for i in range(n):
            total += items[i]
        return total

    def pythonic() -> int:
        items = list(range(1000))
        return sum(items)  # Best!

    t1 = timeit.timeit(with_len_call, number=1000)
    t2 = timeit.timeit(without_len_call, number=1000)
    t3 = timeit.timeit(pythonic, number=1000)
    print(f"len() in loop: {t1:.4f}s")
    print(f"len() cached: {t2:.4f}s")
    print(f"sum() built-in: {t3:.4f}s")
    print()


# =====================
# Memory Optimization
# =====================


def demo_memory() -> None:
    print("=== Memory Optimization ===\n")

    # __slots__ for memory efficiency
    class RegularClass:
        def __init__(self, x: int, y: int) -> None:
            self.x = x
            self.y = y

    class SlottedClass:
        __slots__ = ["x", "y"]

        def __init__(self, x: int, y: int) -> None:
            self.x = x
            self.y = y

    regular = RegularClass(1, 2)
    slotted = SlottedClass(1, 2)

    print(f"Regular has __dict__: {hasattr(regular, '__dict__')}")
    print(f"Slotted has __dict__: {hasattr(slotted, '__dict__')}")
    print(
        f"Regular size: {sys.getsizeof(regular)} + {sys.getsizeof(regular.__dict__)} bytes"
    )
    print(f"Slotted size: {sys.getsizeof(slotted)} bytes")
    print()

    # Generator vs List for memory
    print("--- Generator Memory ---")

    # This would use ~800MB of RAM:
    # big_list = [x for x in range(100_000_000)]

    # This uses almost no RAM:
    # big_gen = (x for x in range(100_000_000))

    print("List of 10M items: ~80MB")
    print("Generator of 10M items: ~100 bytes")
    print()


# =====================
# Caching
# =====================


def demo_caching() -> None:
    print("=== Caching ===\n")

    # Without cache
    call_count = 0

    def fib_slow(n: int) -> int:
        nonlocal call_count
        call_count += 1
        if n < 2:
            return n
        return fib_slow(n - 1) + fib_slow(n - 2)

    call_count = 0
    result = fib_slow(20)
    print(f"fib(20) without cache: {result}, calls: {call_count}")

    # With cache
    @functools.lru_cache(maxsize=None)
    def fib_fast(n: int) -> int:
        if n < 2:
            return n
        return fib_fast(n - 1) + fib_fast(n - 2)

    result = fib_fast(20)
    print(f"fib(20) with cache: {result}")
    print(f"Cache info: {fib_fast.cache_info()}")
    print()


# =====================
# Algorithmic Improvements
# =====================


def demo_algorithmic() -> None:
    print("=== Algorithmic Improvements ===\n")

    # O(n²) vs O(n) for finding duplicates
    data = list(range(1000)) + [500]  # One duplicate

    def find_dup_n2(items: list[int]) -> int | None:
        """O(n²) - nested loops."""
        for i, x in enumerate(items):
            for j, y in enumerate(items):
                if i != j and x == y:
                    return x
        return None

    def find_dup_n(items: list[int]) -> int | None:
        """O(n) - use set."""
        seen: set[int] = set()
        for x in items:
            if x in seen:
                return x
            seen.add(x)
        return None

    t1 = timeit.timeit(lambda: find_dup_n2(data), number=10)
    t2 = timeit.timeit(lambda: find_dup_n(data), number=10)

    print(f"O(n²): {t1:.4f}s")
    print(f"O(n): {t2:.4f}s")
    print(f"Speedup: {t1/t2:.0f}x")
    print()


# =====================
# Summary
# =====================


def print_summary() -> None:
    print("=== Optimization Summary ===\n")
    tips = [
        "1. Profile before optimizing (cProfile, timeit)",
        "2. Use appropriate data structures (set for lookup, deque for queues)",
        "3. Prefer generators for large intermediate results",
        "4. Use ''.join() instead of += for strings",
        "5. Cache expensive computations (lru_cache)",
        "6. Use __slots__ for memory-critical classes",
        "7. Prefer built-in functions (sum, map, filter)",
        "8. Consider algorithmic improvements first",
        "9. Use local variables in hot loops",
        "10. For CPU-bound: consider NumPy, Cython, or multiprocessing",
    ]
    for tip in tips:
        print(tip)


# =====================
# Demo
# =====================

if __name__ == "__main__":
    print("=== Profiling & Optimization Demo ===\n")

    demo_timeit()
    demo_cprofile()
    demo_optimizations()
    demo_memory()
    demo_caching()
    demo_algorithmic()
    print_summary()
