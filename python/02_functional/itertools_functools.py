"""
TOPIC: itertools & functools

USE CASES:
  - Data processing pipelines
  - Lazy evaluation for large datasets
  - Functional programming patterns
  - Combinatorics and permutations

KEY POINTS:
  - itertools: Iterator building blocks (lazy, memory efficient)
  - functools: Higher-order functions (reduce, partial, cache)
  - Compose iterators for complex transformations
  - Most return iterators, not lists (call list() to materialize)
"""

import itertools
import functools
import operator
from typing import Iterable, TypeVar, Callable

T = TypeVar("T")

# =====================
# itertools: Infinite Iterators
# =====================

def demo_infinite() -> None:
    print("=== Infinite Iterators ===")

    # count: infinite counter
    counter = itertools.count(start=10, step=2)
    print(f"count(10, 2): {[next(counter) for _ in range(5)]}")

    # cycle: repeat iterable forever
    colors = itertools.cycle(["red", "green", "blue"])
    print(f"cycle colors: {[next(colors) for _ in range(7)]}")

    # repeat: repeat value
    print(f"repeat('x', 4): {list(itertools.repeat('x', 4))}")
    print()


# =====================
# itertools: Combinatorics
# =====================

def demo_combinatorics() -> None:
    print("=== Combinatorics ===")

    items = ["A", "B", "C"]

    # permutations: all orderings
    print(f"permutations({items}, 2): {list(itertools.permutations(items, 2))}")

    # combinations: unordered subsets
    print(f"combinations({items}, 2): {list(itertools.combinations(items, 2))}")

    # combinations_with_replacement
    print(f"comb_w_replacement({items[:2]}, 2): {list(itertools.combinations_with_replacement(items[:2], 2))}")

    # product: cartesian product
    print(f"product([1,2], ['a','b']): {list(itertools.product([1, 2], ['a', 'b']))}")

    # product with repeat
    print(f"product([0,1], repeat=3): {list(itertools.product([0, 1], repeat=3))}")
    print()


# =====================
# itertools: Filtering
# =====================

def demo_filtering() -> None:
    print("=== Filtering ===")

    data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # takewhile: take while predicate is true
    print(f"takewhile(x < 5): {list(itertools.takewhile(lambda x: x < 5, data))}")

    # dropwhile: skip while predicate is true
    print(f"dropwhile(x < 5): {list(itertools.dropwhile(lambda x: x < 5, data))}")

    # filterfalse: opposite of filter
    print(f"filterfalse(is_even): {list(itertools.filterfalse(lambda x: x % 2 == 0, data))}")

    # compress: filter by selector
    selectors = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    print(f"compress with selectors: {list(itertools.compress(data, selectors))}")
    print()


# =====================
# itertools: Grouping
# =====================

def demo_grouping() -> None:
    print("=== Grouping ===")

    # groupby: group consecutive items (must be sorted!)
    data = [("A", 1), ("A", 2), ("B", 3), ("B", 4), ("A", 5)]
    sorted_data = sorted(data, key=lambda x: x[0])

    print("groupby (sorted by first element):")
    for key, group in itertools.groupby(sorted_data, key=lambda x: x[0]):
        print(f"  {key}: {list(group)}")

    # Practical example: group numbers by tens
    numbers = [3, 15, 22, 27, 31, 38, 45]
    for decade, group in itertools.groupby(numbers, key=lambda x: x // 10):
        print(f"  {decade}0s: {list(group)}")
    print()


# =====================
# itertools: Combining
# =====================

def demo_combining() -> None:
    print("=== Combining ===")

    # chain: flatten iterables
    a = [1, 2, 3]
    b = [4, 5, 6]
    print(f"chain([1,2,3], [4,5,6]): {list(itertools.chain(a, b))}")

    # chain.from_iterable: flatten nested
    nested = [[1, 2], [3, 4], [5, 6]]
    print(f"chain.from_iterable: {list(itertools.chain.from_iterable(nested))}")

    # zip_longest: zip with fill value
    x = [1, 2, 3]
    y = ["a", "b"]
    print(f"zip_longest: {list(itertools.zip_longest(x, y, fillvalue='-'))}")

    # pairwise (Python 3.10+)
    print(f"pairwise([1,2,3,4]): {list(itertools.pairwise([1, 2, 3, 4]))}")
    print()


# =====================
# itertools: Slicing
# =====================

def demo_slicing() -> None:
    print("=== Slicing ===")

    data = range(100)

    # islice: slice an iterator (can't use [] on iterators)
    print(f"islice(range(100), 5, 10): {list(itertools.islice(data, 5, 10))}")
    print(f"islice(range(100), 0, 10, 2): {list(itertools.islice(range(100), 0, 10, 2))}")

    # tee: create independent iterators
    original = iter([1, 2, 3, 4, 5])
    copy1, copy2 = itertools.tee(original, 2)
    print(f"tee copy1: {list(copy1)}")
    print(f"tee copy2: {list(copy2)}")
    print()


# =====================
# functools: reduce
# =====================

def demo_reduce() -> None:
    print("=== functools.reduce ===")

    numbers = [1, 2, 3, 4, 5]

    # Sum (same as sum())
    total = functools.reduce(operator.add, numbers)
    print(f"reduce(add, [1,2,3,4,5]) = {total}")

    # Product
    product = functools.reduce(operator.mul, numbers)
    print(f"reduce(mul, [1,2,3,4,5]) = {product}")

    # With initial value
    total_with_init = functools.reduce(operator.add, numbers, 100)
    print(f"reduce(add, [1,2,3,4,5], 100) = {total_with_init}")

    # Find max (same as max())
    maximum = functools.reduce(lambda a, b: a if a > b else b, numbers)
    print(f"reduce(max) = {maximum}")

    # Flatten nested list
    nested = [[1, 2], [3, 4], [5]]
    flat = functools.reduce(operator.concat, nested)
    print(f"reduce(concat, nested) = {flat}")
    print()


# =====================
# functools: partial
# =====================

def demo_partial() -> None:
    print("=== functools.partial ===")

    def power(base: int, exponent: int) -> int:
        return base ** exponent

    # Create specialized functions
    square = functools.partial(power, exponent=2)
    cube = functools.partial(power, exponent=3)

    print(f"square(5) = {square(5)}")
    print(f"cube(5) = {cube(5)}")

    # Useful with map/filter
    def multiply(x: int, y: int) -> int:
        return x * y

    double = functools.partial(multiply, 2)
    print(f"map(double, [1,2,3,4,5]) = {list(map(double, [1, 2, 3, 4, 5]))}")
    print()


# =====================
# functools: caching
# =====================

def demo_caching() -> None:
    print("=== functools.cache / lru_cache ===")

    call_count = 0

    @functools.lru_cache(maxsize=128)
    def fibonacci(n: int) -> int:
        nonlocal call_count
        call_count += 1
        if n < 2:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    result = fibonacci(30)
    print(f"fibonacci(30) = {result}")
    print(f"Actual calls made: {call_count}")
    print(f"Cache info: {fibonacci.cache_info()}")

    # Clear cache
    fibonacci.cache_clear()
    print()


# =====================
# functools: singledispatch
# =====================

def demo_singledispatch() -> None:
    print("=== functools.singledispatch ===")

    @functools.singledispatch
    def process(arg: object) -> str:
        return f"Generic: {arg}"

    @process.register(int)
    def _(arg: int) -> str:
        return f"Integer: {arg * 2}"

    @process.register(str)
    def _(arg: str) -> str:
        return f"String: {arg.upper()}"

    @process.register(list)
    def _(arg: list) -> str:
        return f"List of {len(arg)} items"

    print(process(42))
    print(process("hello"))
    print(process([1, 2, 3]))
    print(process(3.14))  # Falls back to generic
    print()


# =====================
# functools: total_ordering
# =====================

def demo_total_ordering() -> None:
    print("=== functools.total_ordering ===")

    @functools.total_ordering
    class Version:
        def __init__(self, major: int, minor: int) -> None:
            self.major = major
            self.minor = minor

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, Version):
                return NotImplemented
            return (self.major, self.minor) == (other.major, other.minor)

        def __lt__(self, other: "Version") -> bool:
            return (self.major, self.minor) < (other.major, other.minor)

        def __repr__(self) -> str:
            return f"v{self.major}.{self.minor}"

    # Only defined __eq__ and __lt__, but get all comparison operators!
    v1 = Version(1, 0)
    v2 = Version(2, 0)
    v3 = Version(1, 5)

    versions = [v2, v1, v3]
    print(f"Sorted: {sorted(versions)}")
    print(f"v1 < v3: {v1 < v3}")
    print(f"v2 >= v1: {v2 >= v1}")
    print()


# =====================
# Practical Recipes
# =====================

def chunked(iterable: Iterable[T], size: int) -> Iterable[list[T]]:
    """Split iterable into chunks of given size."""
    it = iter(iterable)
    while chunk := list(itertools.islice(it, size)):
        yield chunk


def sliding_window(iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    """Sliding window of size n."""
    iterators = itertools.tee(iterable, n)
    for i, it in enumerate(iterators):
        for _ in range(i):
            next(it, None)
    return zip(*iterators)


def demo_recipes() -> None:
    print("=== Practical Recipes ===")

    data = list(range(10))

    print(f"chunked({data}, 3): {list(chunked(data, 3))}")
    print(f"sliding_window([0,1,2,3,4], 3): {list(sliding_window(range(5), 3))}")
    print()


# =====================
# Demo
# =====================

if __name__ == "__main__":
    demo_infinite()
    demo_combinatorics()
    demo_filtering()
    demo_grouping()
    demo_combining()
    demo_slicing()
    demo_reduce()
    demo_partial()
    demo_caching()
    demo_singledispatch()
    demo_total_ordering()
    demo_recipes()
