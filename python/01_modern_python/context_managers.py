"""
TOPIC: Context Managers

USE CASES:
  - Resource management (files, connections, locks)
  - Setup/teardown patterns
  - Temporary state changes
  - Transaction-like operations (commit/rollback)

KEY POINTS:
  - 'with' statement ensures cleanup even on exceptions
  - __enter__ returns resource, __exit__ handles cleanup
  - contextlib.contextmanager for generator-based approach
  - Can be nested or combined with ExitStack
  - async context managers for async resources
"""

from contextlib import contextmanager, ExitStack, suppress
import os
import time
import tempfile
from typing import Iterator, Generator, Any

# =====================
# Class-Based Context Manager
# =====================


class Timer:
    """Context manager to time code execution."""

    def __init__(self, name: str = "Block") -> None:
        self.name = name
        self.start: float = 0
        self.elapsed: float = 0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self  # Returned value bound to 'as' variable

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        self.elapsed = time.perf_counter() - self.start
        print(f"{self.name} took {self.elapsed:.4f}s")
        return False  # Don't suppress exceptions


class DatabaseConnection:
    """Simulated database connection with transaction."""

    def __init__(self, db_name: str) -> None:
        self.db_name = db_name
        self.connected = False
        self.in_transaction = False

    def __enter__(self) -> "DatabaseConnection":
        print(f"Connecting to {self.db_name}...")
        self.connected = True
        self.in_transaction = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        if exc_type is not None:
            print(f"Rolling back due to: {exc_val}")
            self.in_transaction = False
        else:
            print("Committing transaction")
        print(f"Disconnecting from {self.db_name}")
        self.connected = False
        return False  # Re-raise exceptions

    def execute(self, query: str) -> None:
        if not self.connected:
            raise RuntimeError("Not connected")
        print(f"Executing: {query}")


# =====================
# Generator-Based (contextlib)
# =====================


@contextmanager
def timer(name: str = "Block") -> Generator[None, None, None]:
    """Same as Timer class but using generator."""
    start = time.perf_counter()
    try:
        yield  # Code in 'with' block runs here
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name} took {elapsed:.4f}s")


@contextmanager
def temporary_directory() -> Generator[str, None, None]:
    """Create and cleanup a temporary directory."""
    import shutil

    path = tempfile.mkdtemp()
    print(f"Created temp dir: {path}")
    try:
        yield path
    finally:
        shutil.rmtree(path)
        print(f"Removed temp dir: {path}")


@contextmanager
def change_directory(path: str) -> Generator[None, None, None]:
    """Temporarily change working directory."""
    original = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


@contextmanager
def suppress_output() -> Generator[None, None, None]:
    """Suppress stdout temporarily."""
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


# =====================
# Exception Handling in __exit__
# =====================


class ErrorHandler:
    """Demonstrates exception handling in context manager."""

    def __init__(self, suppress_errors: bool = False) -> None:
        self.suppress_errors = suppress_errors
        self.error: Exception | None = None

    def __enter__(self) -> "ErrorHandler":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        if exc_val is not None:
            self.error = exc_val  # type: ignore
            print(f"Caught error: {exc_val}")
        # Return True to suppress the exception
        return self.suppress_errors


# =====================
# ExitStack (Dynamic Context Managers)
# =====================


def process_multiple_files(filenames: list[str]) -> None:
    """Open multiple files dynamically."""
    with ExitStack() as stack:
        files = [stack.enter_context(open(f, "w")) for f in filenames]
        for i, f in enumerate(files):
            f.write(f"Content for file {i}\n")
    # All files automatically closed


@contextmanager
def managed_resources() -> Generator[list[Any], None, None]:
    """Manage multiple resources with ExitStack."""
    resources: list[Any] = []
    with ExitStack() as stack:
        # Add cleanup callbacks
        stack.callback(lambda: print("Final cleanup"))

        # Resources added here will be cleaned up
        yield resources

        # Cleanup happens in reverse order


# =====================
# Reentrant Context Managers
# =====================


class ReentrantLock:
    """Context manager that can be entered multiple times."""

    def __init__(self) -> None:
        self.count = 0

    def __enter__(self) -> "ReentrantLock":
        self.count += 1
        if self.count == 1:
            print("Lock acquired")
        else:
            print(f"Lock re-entered (depth: {self.count})")
        return self

    def __exit__(self, *args: Any) -> bool:
        self.count -= 1
        if self.count == 0:
            print("Lock released")
        else:
            print(f"Lock depth now: {self.count}")
        return False


# =====================
# Async Context Managers
# =====================


class AsyncDatabase:
    """Async context manager example."""

    async def __aenter__(self) -> "AsyncDatabase":
        print("Async: Connecting...")
        # await asyncio.sleep(0.1)  # Simulate async connection
        return self

    async def __aexit__(self, *args: Any) -> bool:
        print("Async: Disconnecting...")
        return False


# Or using contextlib:
# from contextlib import asynccontextmanager
# @asynccontextmanager
# async def async_timer():
#     start = time.perf_counter()
#     try:
#         yield
#     finally:
#         print(f"Took {time.perf_counter() - start:.4f}s")


# =====================
# Demo
# =====================

if __name__ == "__main__":
    print("=== Context Managers Demo ===\n")

    # Class-based timer
    print("--- Class-Based Timer ---")
    with Timer("Sleep test") as t:
        time.sleep(0.1)
    print(f"Elapsed stored: {t.elapsed:.4f}s")
    print()

    # Generator-based timer
    print("--- Generator-Based Timer ---")
    with timer("Computation"):
        sum(range(1_000_000))
    print()

    # Database simulation
    print("--- Database Transaction (success) ---")
    with DatabaseConnection("mydb") as db:
        db.execute("INSERT INTO users VALUES (1, 'Alice')")
        db.execute("UPDATE users SET name = 'Bob'")
    print()

    print("--- Database Transaction (failure) ---")
    try:
        with DatabaseConnection("mydb") as db:
            db.execute("INSERT INTO users VALUES (2, 'Charlie')")
            raise ValueError("Something went wrong!")
    except ValueError:
        print("Exception propagated\n")

    # Temporary directory
    print("--- Temporary Directory ---")
    with temporary_directory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.txt")
        with open(filepath, "w") as f:
            f.write("Hello!")
        print(f"File exists: {os.path.exists(filepath)}")
    print()

    # Error suppression
    print("--- Error Suppression ---")
    with ErrorHandler(suppress_errors=True) as handler:
        raise RuntimeError("This error is suppressed")
    print(f"Stored error: {handler.error}")
    print()

    # Built-in suppress
    print("--- contextlib.suppress ---")
    with suppress(FileNotFoundError):
        os.remove("nonexistent_file.txt")
    print("No error raised!")
    print()

    # Reentrant
    print("--- Reentrant Lock ---")
    lock = ReentrantLock()
    with lock:
        print("Outer block")
        with lock:
            print("Inner block")
    print()

    # Nested contexts
    print("--- Nested Contexts ---")
    with Timer("Outer"), timer("Inner"):
        time.sleep(0.05)
