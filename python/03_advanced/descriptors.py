"""
TOPIC: Descriptors

USE CASES:
  - Property-like behavior with reuse
  - Type validation, lazy loading
  - ORM field definitions
  - Computed attributes with caching

KEY POINTS:
  - Descriptor = object with __get__, __set__, __delete__
  - Non-data descriptor: only __get__ (e.g., methods)
  - Data descriptor: __get__ + __set__ (e.g., property)
  - __set_name__ gets attribute name automatically
  - Descriptors power: property, classmethod, staticmethod
"""

from typing import Any, TypeVar, Generic, Callable, overload
from weakref import WeakKeyDictionary
import time

T = TypeVar("T")

# =====================
# Basic Descriptor
# =====================


class Verbose:
    """Descriptor that logs access."""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.private_name = f"_desc_{name}"

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        value = getattr(obj, self.private_name, None)
        print(f"Getting {self.name}: {value}")
        return value

    def __set__(self, obj: Any, value: Any) -> None:
        print(f"Setting {self.name} = {value}")
        setattr(obj, self.private_name, value)


class Example:
    x = Verbose()
    y = Verbose()


# =====================
# Typed Descriptor
# =====================


class Typed(Generic[T]):
    """Descriptor that enforces type."""

    def __init__(self, expected_type: type) -> None:
        self.expected_type = expected_type
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @overload
    def __get__(self, obj: None, objtype: type) -> "Typed[T]": ...
    @overload
    def __get__(self, obj: object, objtype: type) -> T: ...

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return obj.__dict__.get(self.name)

    def __set__(self, obj: Any, value: T) -> None:
        if not isinstance(value, self.expected_type):
            raise TypeError(
                f"{self.name} must be {self.expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
        obj.__dict__[self.name] = value


class Person:
    name = Typed(str)
    age = Typed(int)

    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age


# =====================
# Validated Descriptor
# =====================


class Validated:
    """Descriptor with custom validation."""

    def __init__(
        self,
        validator: Callable[[Any], bool],
        error_msg: str = "Validation failed",
    ) -> None:
        self.validator = validator
        self.error_msg = error_msg
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return obj.__dict__.get(self.name)

    def __set__(self, obj: Any, value: Any) -> None:
        if not self.validator(value):
            raise ValueError(f"{self.name}: {self.error_msg}")
        obj.__dict__[self.name] = value


class Product:
    name = Validated(
        lambda x: isinstance(x, str) and len(x) > 0,
        "must be non-empty string",
    )
    price = Validated(
        lambda x: isinstance(x, (int, float)) and x > 0,
        "must be positive number",
    )
    quantity = Validated(
        lambda x: isinstance(x, int) and x >= 0,
        "must be non-negative integer",
    )

    def __init__(self, name: str, price: float, quantity: int = 0) -> None:
        self.name = name
        self.price = price
        self.quantity = quantity


# =====================
# Lazy Property (Cached)
# =====================


class LazyProperty:
    """Compute once, then cache result."""

    def __init__(self, func: Callable[[Any], T]) -> None:
        self.func = func
        self.name = func.__name__

    def __get__(self, obj: Any, objtype: type | None = None) -> T:
        if obj is None:
            return self  # type: ignore
        # Compute and cache in instance __dict__
        value = self.func(obj)
        obj.__dict__[self.name] = value  # Replace descriptor access!
        return value


class DataLoader:
    def __init__(self, path: str) -> None:
        self.path = path

    @LazyProperty
    def data(self) -> list[int]:
        print(f"Loading data from {self.path}... (expensive!)")
        time.sleep(0.1)  # Simulate expensive operation
        return [1, 2, 3, 4, 5]


# =====================
# Non-Data Descriptor (like methods)
# =====================


class Method:
    """Simulate how methods work."""

    def __init__(self, func: Callable) -> None:
        self.func = func

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self.func  # Unbound

        # Return bound method
        def bound_method(*args: Any, **kwargs: Any) -> Any:
            return self.func(obj, *args, **kwargs)

        return bound_method


class MyClass:
    @Method
    def greet(self, name: str) -> str:
        return f"Hello, {name}!"


# =====================
# Descriptor with WeakRef (no memory leak)
# =====================


class WeakTyped:
    """Type-checked descriptor using WeakKeyDictionary."""

    def __init__(self, expected_type: type) -> None:
        self.expected_type = expected_type
        self.data: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return self.data.get(obj)

    def __set__(self, obj: Any, value: Any) -> None:
        if not isinstance(value, self.expected_type):
            raise TypeError(
                f"{self.name} must be {self.expected_type.__name__}"
            )
        self.data[obj] = value


# =====================
# Computed Property
# =====================


class Computed:
    """Property that depends on other attributes."""

    def __init__(self, *depends_on: str, compute: Callable[..., T]) -> None:
        self.depends_on = depends_on
        self.compute = compute
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> T:
        if obj is None:
            return self  # type: ignore
        args = [getattr(obj, dep) for dep in self.depends_on]
        return self.compute(*args)


class Rectangle:
    width = Typed(float)
    height = Typed(float)
    area = Computed("width", "height", compute=lambda w, h: w * h)
    perimeter = Computed("width", "height", compute=lambda w, h: 2 * (w + h))

    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height


# =====================
# How property() Works
# =====================


class PropertyLike:
    """Simplified implementation of property()."""

    def __init__(
        self,
        fget: Callable | None = None,
        fset: Callable | None = None,
        fdel: Callable | None = None,
        doc: str | None = None,
    ) -> None:
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc or (fget.__doc__ if fget else None)

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(obj)

    def __set__(self, obj: Any, value: Any) -> None:
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(obj, value)

    def __delete__(self, obj: Any) -> None:
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(obj)

    def getter(self, fget: Callable) -> "PropertyLike":
        return PropertyLike(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset: Callable) -> "PropertyLike":
        return PropertyLike(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel: Callable) -> "PropertyLike":
        return PropertyLike(self.fget, self.fset, fdel, self.__doc__)


# =====================
# Demo
# =====================

if __name__ == "__main__":
    print("=== Descriptors Demo ===\n")

    # Verbose descriptor
    print("--- Verbose Descriptor ---")
    ex = Example()
    ex.x = 10
    ex.y = 20
    _ = ex.x
    print()

    # Typed descriptor
    print("--- Typed Descriptor ---")
    p = Person("Alice", 30)
    print(f"Person: {p.name}, {p.age}")
    try:
        p.age = "thirty"  # type: ignore
    except TypeError as e:
        print(f"Type error: {e}")
    print()

    # Validated descriptor
    print("--- Validated Descriptor ---")
    prod = Product("Widget", 9.99, 100)
    print(f"Product: {prod.name}, ${prod.price}, qty={prod.quantity}")
    try:
        prod.price = -5
    except ValueError as e:
        print(f"Validation error: {e}")
    print()

    # Lazy property
    print("--- Lazy Property ---")
    loader = DataLoader("/path/to/data")
    print("First access:")
    print(f"  data = {loader.data}")
    print("Second access (cached):")
    print(f"  data = {loader.data}")
    print()

    # Computed property
    print("--- Computed Property ---")
    rect = Rectangle(5.0, 3.0)
    print(f"Rectangle: {rect.width} x {rect.height}")
    print(f"  area = {rect.area}")
    print(f"  perimeter = {rect.perimeter}")
    rect.width = 10.0
    print(f"After width=10: area = {rect.area}")
    print()

    # Method descriptor
    print("--- Method Descriptor ---")
    obj = MyClass()
    print(obj.greet("World"))
