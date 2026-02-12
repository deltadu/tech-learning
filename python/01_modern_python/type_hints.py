"""
TOPIC: Type Hints & Static Typing

USE CASES:
  - Large codebases: Catch bugs before runtime
  - IDE support: Better autocomplete and refactoring
  - Documentation: Types serve as inline docs
  - Team collaboration: Clear interfaces

KEY POINTS:
  - Type hints are optional but increasingly standard
  - Use mypy or pyright for static checking
  - typing module for complex types (List, Dict, Optional, Union)
  - Python 3.10+: Use | instead of Union, list instead of List
"""

from typing import Optional, Union, Callable, TypeVar, Generic
from collections.abc import Sequence, Mapping, Iterator
from dataclasses import dataclass

# =====================
# Basic Type Hints
# =====================


def greet(name: str, times: int = 1) -> str:
    return (f"Hello, {name}! " * times).strip()


def process_items(items: list[int]) -> list[int]:  # Python 3.9+
    return [x * 2 for x in items]


# Return None explicitly
def log_message(msg: str) -> None:
    print(f"[LOG] {msg}")


# =====================
# Optional & Union
# =====================


# Optional[X] = X | None
def find_user(user_id: int) -> Optional[str]:
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)  # Returns None if not found


# Union (pre-3.10) or | (3.10+)
def parse_input(value: int | str) -> int:
    if isinstance(value, str):
        return int(value)
    return value


# =====================
# Collections
# =====================


def process_mapping(data: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(data.items(), key=lambda x: x[1])


def first_or_default(items: Sequence[str], default: str = "") -> str:
    """Sequence works for list, tuple, str, etc."""
    return items[0] if items else default


# =====================
# Callable Types
# =====================


# Function that takes a function as argument
def apply_twice(func: Callable[[int], int], value: int) -> int:
    return func(func(value))


def double(x: int) -> int:
    return x * 2


# Higher-order function with complex signature
Predicate = Callable[[int], bool]


def filter_items(items: list[int], predicate: Predicate) -> list[int]:
    return [x for x in items if predicate(x)]


# =====================
# TypeVar (Generics)
# =====================

T = TypeVar("T")


def first(items: list[T]) -> T | None:
    """Generic function: works with any type."""
    return items[0] if items else None


# Bounded TypeVar
Numeric = TypeVar("Numeric", int, float)


def add_values(a: Numeric, b: Numeric) -> Numeric:
    return a + b


# =====================
# Generic Classes
# =====================

T = TypeVar("T")


class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

    def is_empty(self) -> bool:
        return len(self._items) == 0


# =====================
# Type Aliases
# =====================

UserId = int
UserName = str
UserMap = dict[UserId, UserName]


def get_user_name(users: UserMap, user_id: UserId) -> UserName | None:
    return users.get(user_id)


# Complex type alias
JsonValue = (
    str
    | int
    | float
    | bool
    | None
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)


# =====================
# Protocols (Structural Subtyping)
# =====================

from typing import Protocol


class Drawable(Protocol):
    def draw(self) -> None: ...


class Circle:
    def draw(self) -> None:
        print("Drawing circle")


class Square:
    def draw(self) -> None:
        print("Drawing square")


def render(shape: Drawable) -> None:
    """Accepts any object with a draw() method."""
    shape.draw()


# =====================
# Literal Types
# =====================

from typing import Literal


def set_mode(mode: Literal["read", "write", "append"]) -> None:
    print(f"Mode set to: {mode}")


# =====================
# TypedDict
# =====================

from typing import TypedDict


class UserDict(TypedDict):
    name: str
    age: int
    email: str


def process_user(user: UserDict) -> str:
    return f"{user['name']} ({user['age']})"


# =====================
# Self Type (Python 3.11+)
# =====================

from typing import Self


class Builder:
    def __init__(self) -> None:
        self.value = 0

    def add(self, n: int) -> Self:
        self.value += n
        return self

    def multiply(self, n: int) -> Self:
        self.value *= n
        return self


# =====================
# Demo
# =====================

if __name__ == "__main__":
    print("=== Type Hints Demo ===\n")

    # Basic
    print(greet("World", 2))

    # Optional
    print(f"User 1: {find_user(1)}")
    print(f"User 99: {find_user(99)}")

    # Callable
    print(f"apply_twice(double, 3) = {apply_twice(double, 3)}")

    # Generics
    print(f"first([1, 2, 3]) = {first([1, 2, 3])}")
    print(f"first(['a', 'b']) = {first(['a', 'b'])}")

    # Generic class
    stack: Stack[str] = Stack()
    stack.push("hello")
    stack.push("world")
    print(f"Stack pop: {stack.pop()}")

    # Protocol
    render(Circle())
    render(Square())

    # Builder pattern with Self
    result = Builder().add(5).multiply(3).add(2).value
    print(f"Builder result: {result}")

    print("\n✅ Run 'mypy type_hints.py' to check types statically")
