"""
TOPIC: Decorators

USE CASES:
  - Logging, timing, caching (cross-cutting concerns)
  - Authentication/authorization checks
  - Retry logic, rate limiting
  - Registration patterns (Flask routes, pytest fixtures)

KEY POINTS:
  - Decorator = function that wraps another function
  - @decorator is syntax sugar for func = decorator(func)
  - Use functools.wraps to preserve metadata
  - Decorators can take arguments (triple-nested)
  - Class decorators modify/wrap classes
"""

import functools
import time
from typing import Callable, TypeVar, ParamSpec, Any

P = ParamSpec("P")
R = TypeVar("R")

# =====================
# Basic Decorator
# =====================

def simple_logger(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)  # Preserves __name__, __doc__, etc.
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper


@simple_logger
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# =====================
# Timing Decorator
# =====================

def timer(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper


@timer
def slow_function() -> str:
    time.sleep(0.1)
    return "done"


# =====================
# Decorator with Arguments
# =====================

def repeat(times: int) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that repeats function call."""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            result = None
            for _ in range(times):
                result = func(*args, **kwargs)
            return result  # type: ignore
        return wrapper
    return decorator


@repeat(times=3)
def say_hello(name: str) -> None:
    print(f"Hello, {name}!")


# =====================
# Retry Decorator
# =====================

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,)
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry on failure with exponential backoff."""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    wait = delay * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
            raise last_exception  # type: ignore
        return wrapper
    return decorator


# =====================
# Memoization (Caching)
# =====================

def memoize(func: Callable[P, R]) -> Callable[P, R]:
    """Simple memoization decorator."""
    cache: dict[tuple, R] = {}

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper


# Better: use functools.lru_cache or functools.cache (3.9+)
@functools.lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# =====================
# Decorator that Works with/without Parentheses
# =====================

def debug(_func: Callable[P, R] | None = None, *, prefix: str = "DEBUG") -> Callable:
    """Can be used as @debug or @debug(prefix='INFO')."""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            print(f"[{prefix}] {func.__name__}({args}, {kwargs})")
            return func(*args, **kwargs)
        return wrapper

    if _func is not None:
        return decorator(_func)
    return decorator


@debug  # Without parentheses
def func1() -> str:
    return "func1"


@debug(prefix="INFO")  # With parentheses
def func2() -> str:
    return "func2"


# =====================
# Class as Decorator
# =====================

class CountCalls:
    """Decorator that counts function calls."""

    def __init__(self, func: Callable) -> None:
        functools.update_wrapper(self, func)
        self.func = func
        self.count = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.count += 1
        return self.func(*args, **kwargs)


@CountCalls
def greet(name: str) -> str:
    return f"Hello, {name}"


# =====================
# Class Decorator
# =====================

def singleton(cls: type) -> type:
    """Make a class a singleton."""
    instances: dict[type, Any] = {}

    @functools.wraps(cls)
    def get_instance(*args: Any, **kwargs: Any) -> Any:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance  # type: ignore


@singleton
class Database:
    def __init__(self) -> None:
        print("Initializing database connection...")
        self.connected = True


# =====================
# Method Decorators
# =====================

def validate_positive(func: Callable) -> Callable:
    """Validate that first numeric argument is positive."""
    @functools.wraps(func)
    def wrapper(self: Any, value: float, *args: Any, **kwargs: Any) -> Any:
        if value <= 0:
            raise ValueError(f"{func.__name__}: value must be positive, got {value}")
        return func(self, value, *args, **kwargs)
    return wrapper


class BankAccount:
    def __init__(self, balance: float = 0) -> None:
        self.balance = balance

    @validate_positive
    def deposit(self, amount: float) -> None:
        self.balance += amount

    @validate_positive
    def withdraw(self, amount: float) -> None:
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount


# =====================
# Stacking Decorators
# =====================

@timer
@simple_logger
def compute(x: int) -> int:
    """Compute something."""
    return x ** 2


# Equivalent to: compute = timer(simple_logger(compute))
# Order matters! timer runs first, then simple_logger


# =====================
# Demo
# =====================

if __name__ == "__main__":
    print("=== Decorators Demo ===\n")

    # Basic decorator
    print("--- Basic Logger ---")
    result = add(2, 3)
    print()

    # Timer
    print("--- Timer ---")
    slow_function()
    print()

    # Repeat
    print("--- Repeat ---")
    say_hello("World")
    print()

    # Memoization
    print("--- Memoization (fibonacci) ---")
    print(f"fib(30) = {fibonacci(30)}")
    print(f"Cache info: {fibonacci.cache_info()}")
    print()

    # Debug with/without args
    print("--- Debug Decorator ---")
    func1()
    func2()
    print()

    # Class as decorator
    print("--- CountCalls ---")
    greet("Alice")
    greet("Bob")
    greet("Charlie")
    print(f"greet was called {greet.count} times")
    print()

    # Singleton
    print("--- Singleton ---")
    db1 = Database()
    db2 = Database()
    print(f"db1 is db2: {db1 is db2}")
    print()

    # Method decorator
    print("--- Method Decorator ---")
    account = BankAccount(100)
    account.deposit(50)
    print(f"Balance: {account.balance}")
    try:
        account.deposit(-10)
    except ValueError as e:
        print(f"Error: {e}")
    print()

    # Stacked decorators
    print("--- Stacked Decorators ---")
    compute(5)
