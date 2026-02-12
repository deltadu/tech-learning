"""
TOPIC: Metaclasses & Class Creation

USE CASES:
  - ORMs (SQLAlchemy, Django models)
  - API frameworks (validation, serialization)
  - Plugin systems, registries
  - Enforcing coding patterns

KEY POINTS:
  - Metaclass = class of a class
  - __new__ creates class, __init__ initializes it
  - __call__ controls instance creation
  - Often replaced by decorators or __init_subclass__
  - Use sparingly: "If you wonder if you need metaclasses, you don't"
"""

from typing import Any, TypeVar, Callable

T = TypeVar("T")

# =====================
# Classes are Objects
# =====================


def demo_class_as_object() -> None:
    print("=== Classes are Objects ===")

    class MyClass:
        x = 10

    # Class is instance of type
    print(f"type(MyClass) = {type(MyClass)}")
    print(f"type(int) = {type(int)}")
    print(f"isinstance(MyClass, type) = {isinstance(MyClass, type)}")

    # Create class dynamically with type()
    DynamicClass = type(
        "DynamicClass", (object,), {"y": 20, "greet": lambda self: "Hi!"}
    )
    obj = DynamicClass()
    print(f"DynamicClass.y = {obj.y}")
    print(f"obj.greet() = {obj.greet()}")
    print()


# =====================
# Basic Metaclass
# =====================


class LoggingMeta(type):
    """Metaclass that logs class creation."""

    def __new__(
        mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any]
    ) -> type:
        print(f"Creating class: {name}")
        print(f"  Bases: {bases}")
        print(f"  Attributes: {list(namespace.keys())}")
        return super().__new__(mcs, name, bases, namespace)


class MyService(metaclass=LoggingMeta):
    def serve(self) -> str:
        return "Serving..."


# =====================
# Registry Pattern
# =====================


class PluginRegistry(type):
    """Auto-register subclasses."""

    plugins: dict[str, type] = {}

    def __new__(
        mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any]
    ) -> type:
        cls = super().__new__(mcs, name, bases, namespace)
        if bases:  # Don't register base class
            mcs.plugins[name] = cls
            print(f"Registered plugin: {name}")
        return cls


class Plugin(metaclass=PluginRegistry):
    """Base class for plugins."""

    pass


class AudioPlugin(Plugin):
    def process(self) -> str:
        return "Processing audio"


class VideoPlugin(Plugin):
    def process(self) -> str:
        return "Processing video"


# =====================
# Singleton Metaclass
# =====================


class SingletonMeta(type):
    """Metaclass that makes classes singletons."""

    _instances: dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Database(metaclass=SingletonMeta):
    def __init__(self) -> None:
        print("Database initialized")
        self.connected = True


# =====================
# Validation Metaclass
# =====================


class ValidatedMeta(type):
    """Ensure subclasses implement required methods."""

    required_methods: list[str] = ["validate", "process"]

    def __new__(
        mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any]
    ) -> type:
        # Skip base class check
        if bases:
            for method in mcs.required_methods:
                if method not in namespace:
                    raise TypeError(
                        f"Class {name} must implement {method}()"
                    )
        return super().__new__(mcs, name, bases, namespace)


class Processor(metaclass=ValidatedMeta):
    """Base processor class."""

    pass


class DataProcessor(Processor):
    def validate(self) -> bool:
        return True

    def process(self) -> str:
        return "Processing data"


# This would raise TypeError:
# class BadProcessor(Processor):
#     def validate(self) -> bool:
#         return True
#     # Missing process()!


# =====================
# Attribute Auto-Documentation
# =====================


class DocumentedMeta(type):
    """Auto-add docstring listing class attributes."""

    def __new__(
        mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any]
    ) -> type:
        # Find class-level attributes
        attrs = {
            k: v
            for k, v in namespace.items()
            if not k.startswith("_") and not callable(v)
        }

        if attrs:
            doc = namespace.get("__doc__", "") or ""
            attr_doc = "\n\nAttributes:\n" + "\n".join(
                f"  {k}: {type(v).__name__} = {v!r}"
                for k, v in attrs.items()
            )
            namespace["__doc__"] = doc + attr_doc

        return super().__new__(mcs, name, bases, namespace)


class Config(metaclass=DocumentedMeta):
    """Application configuration."""

    host = "localhost"
    port = 8080
    debug = True


# =====================
# __init_subclass__ (Python 3.6+, simpler alternative)
# =====================


class PluginBase:
    """Base class with subclass hook (no metaclass needed!)."""

    registry: dict[str, type] = {}

    def __init_subclass__(
        cls, plugin_name: str | None = None, **kwargs: Any
    ) -> None:
        super().__init_subclass__(**kwargs)
        name = plugin_name or cls.__name__
        PluginBase.registry[name] = cls
        print(f"Registered via __init_subclass__: {name}")


class MyPlugin(PluginBase, plugin_name="awesome_plugin"):
    pass


class AnotherPlugin(PluginBase):
    pass


# =====================
# __class_getitem__ (Generic-like syntax)
# =====================


class Container:
    """Allow Container[T] syntax."""

    def __class_getitem__(cls, item: type) -> str:
        return f"Container of {item.__name__}"


# =====================
# Practical: ORM-style Field Definition
# =====================


class Field:
    """Descriptor for ORM-like fields."""

    def __init__(self, field_type: type) -> None:
        self.field_type = field_type
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return obj.__dict__.get(self.name)

    def __set__(self, obj: Any, value: Any) -> None:
        if not isinstance(value, self.field_type):
            raise TypeError(
                f"{self.name} must be {self.field_type.__name__}, "
                f"got {type(value).__name__}"
            )
        obj.__dict__[self.name] = value


class ModelMeta(type):
    """Collect fields and create schema."""

    def __new__(
        mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any]
    ) -> type:
        fields = {k: v for k, v in namespace.items() if isinstance(v, Field)}
        namespace["_fields"] = fields
        return super().__new__(mcs, name, bases, namespace)


class Model(metaclass=ModelMeta):
    """Base model class."""

    _fields: dict[str, Field]

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={getattr(self, k)!r}" for k in self._fields)
        return f"{self.__class__.__name__}({attrs})"


class User(Model):
    name = Field(str)
    age = Field(int)


# =====================
# Demo
# =====================

if __name__ == "__main__":
    print("=== Metaclasses Demo ===\n")

    demo_class_as_object()

    # Registry pattern
    print("=== Plugin Registry ===")
    print(f"Registered plugins: {list(PluginRegistry.plugins.keys())}")
    print()

    # Singleton
    print("=== Singleton ===")
    db1 = Database()
    db2 = Database()
    print(f"db1 is db2: {db1 is db2}")
    print()

    # Documented
    print("=== Auto-Documentation ===")
    print(f"Config.__doc__:\n{Config.__doc__}")
    print()

    # __init_subclass__
    print("=== __init_subclass__ Registry ===")
    print(f"Plugins: {list(PluginBase.registry.keys())}")
    print()

    # Container[T]
    print("=== __class_getitem__ ===")
    print(Container[int])
    print()

    # ORM-style model
    print("=== ORM-Style Model ===")
    user = User()
    user.name = "Alice"
    user.age = 30
    print(user)

    try:
        user.age = "thirty"  # Type error!
    except TypeError as e:
        print(f"Type error: {e}")
