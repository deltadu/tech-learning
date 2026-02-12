"""
TOPIC: Dataclasses & Modern Data Structures

USE CASES:
  - DTOs (Data Transfer Objects)
  - Configuration objects
  - Replacing namedtuples with more features
  - Immutable data structures

KEY POINTS:
  - @dataclass auto-generates __init__, __repr__, __eq__
  - frozen=True for immutability
  - field() for defaults, factories, metadata
  - __post_init__ for validation/derived fields
  - slots=True (3.10+) for memory efficiency
"""

from dataclasses import dataclass, field, asdict, astuple, replace
from typing import ClassVar
import json

# =====================
# Basic Dataclass
# =====================


@dataclass
class Point:
    x: float
    y: float

    def distance_from_origin(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5


# =====================
# Default Values & Field
# =====================


@dataclass
class User:
    name: str
    email: str
    age: int = 0  # Default value
    tags: list[str] = field(default_factory=list)  # Mutable default
    _id: int = field(default=0, repr=False)  # Exclude from repr

    # Class variable (not a field)
    MAX_AGE: ClassVar[int] = 150


# =====================
# Frozen (Immutable)
# =====================


@dataclass(frozen=True)
class ImmutableConfig:
    host: str
    port: int
    debug: bool = False

    # Frozen dataclasses are hashable (can be dict keys)


# =====================
# Post-Init Processing
# =====================


@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)  # Computed, not in __init__

    def __post_init__(self) -> None:
        self.area = self.width * self.height
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Dimensions must be positive")


# =====================
# Inheritance
# =====================


@dataclass
class Person:
    name: str
    age: int


@dataclass
class Employee(Person):
    employee_id: str
    department: str = "Engineering"


# =====================
# Slots (Memory Efficient, Python 3.10+)
# =====================


@dataclass(slots=True)
class CompactPoint:
    x: float
    y: float
    # Uses __slots__ instead of __dict__, ~30% less memory


# =====================
# Ordering
# =====================


@dataclass(order=True)
class Version:
    # Fields are compared in order defined
    sort_index: tuple[int, int, int] = field(init=False, repr=False)
    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        self.sort_index = (self.major, self.minor, self.patch)


# =====================
# Conversion Utilities
# =====================


@dataclass
class Config:
    name: str
    value: int
    enabled: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        return cls(**data)


# =====================
# Field Metadata
# =====================


@dataclass
class Product:
    name: str
    price: float = field(metadata={"unit": "USD", "min": 0})
    quantity: int = field(default=0, metadata={"min": 0, "max": 1000})


# =====================
# KW_ONLY (Python 3.10+)
# =====================


@dataclass
class Request:
    url: str
    method: str = "GET"
    # Everything after this must be keyword-only
    _: dataclass_field = field(default=None, repr=False)
    timeout: int = 30
    headers: dict[str, str] = field(default_factory=dict)


# For Python 3.10+, use kw_only=True instead:
# @dataclass(kw_only=True) or field(kw_only=True)


# =====================
# Replace (Copy with Changes)
# =====================


@dataclass(frozen=True)
class Settings:
    theme: str
    font_size: int
    language: str = "en"


# =====================
# Demo
# =====================

if __name__ == "__main__":
    print("=== Dataclasses Demo ===\n")

    # Basic
    p = Point(3, 4)
    print(f"Point: {p}")
    print(f"Distance: {p.distance_from_origin()}")

    # Auto-generated __eq__
    p2 = Point(3, 4)
    print(f"p == p2: {p == p2}")

    print()

    # Default factory for mutable
    u1 = User("Alice", "alice@example.com")
    u2 = User("Bob", "bob@example.com")
    u1.tags.append("admin")
    print(f"u1.tags: {u1.tags}")
    print(f"u2.tags: {u2.tags}")  # Empty! Not shared.

    print()

    # Frozen
    config = ImmutableConfig("localhost", 8080)
    print(f"Config: {config}")
    # config.port = 9000  # Would raise FrozenInstanceError

    # Frozen can be dict keys
    cache: dict[ImmutableConfig, str] = {config: "cached_value"}
    print(f"Cache lookup: {cache[config]}")

    print()

    # Post-init
    rect = Rectangle(5, 3)
    print(f"Rectangle: {rect}")
    print(f"Area (computed): {rect.area}")

    print()

    # Inheritance
    emp = Employee("Charlie", 30, "E123")
    print(f"Employee: {emp}")

    print()

    # Ordering
    versions = [Version(2, 0, 0), Version(1, 9, 5), Version(1, 10, 0)]
    print(f"Sorted versions: {sorted(versions)}")

    print()

    # Conversion
    cfg = Config("feature_flag", 42)
    print(f"As dict: {asdict(cfg)}")
    print(f"As tuple: {astuple(cfg)}")
    print(f"As JSON: {cfg.to_json()}")

    print()

    # Replace (immutable update)
    settings = Settings("dark", 14)
    new_settings = replace(settings, font_size=16)
    print(f"Original: {settings}")
    print(f"Updated: {new_settings}")

    print()

    # Field metadata
    prod = Product("Widget", 9.99, 100)
    for f in prod.__dataclass_fields__.values():
        if f.metadata:
            print(f"Field '{f.name}' metadata: {f.metadata}")


# Define this to avoid the annotation error
from dataclasses import field as dataclass_field
