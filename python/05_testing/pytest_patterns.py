"""
TOPIC: Testing with pytest

USE CASES:
  - Unit testing
  - Integration testing
  - Parametrized tests
  - Mocking external dependencies

KEY POINTS:
  - pytest discovers tests automatically (test_*.py, *_test.py)
  - Simple assert statements (no self.assertEqual)
  - Fixtures for setup/teardown
  - Parametrize for multiple test cases
  - Markers for categorizing tests
"""

import pytest
from typing import Any
from dataclasses import dataclass
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# =====================
# Code to Test
# =====================

@dataclass
class User:
    name: str
    email: str
    age: int

    def is_adult(self) -> bool:
        return self.age >= 18

    def greeting(self) -> str:
        return f"Hello, {self.name}!"


class UserService:
    def __init__(self, db: Any) -> None:
        self.db = db

    def get_user(self, user_id: int) -> User | None:
        data = self.db.fetch(user_id)
        if data:
            return User(**data)
        return None

    def create_user(self, name: str, email: str, age: int) -> User:
        user = User(name=name, email=email, age=age)
        self.db.save(user)
        return user


def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def read_config(filepath: str) -> dict[str, str]:
    config: dict[str, str] = {}
    with open(filepath) as f:
        for line in f:
            key, value = line.strip().split("=")
            config[key] = value
    return config


# =====================
# Basic Tests
# =====================

def test_user_is_adult() -> None:
    """Basic test with assert."""
    adult = User("Alice", "alice@test.com", 25)
    child = User("Bob", "bob@test.com", 10)

    assert adult.is_adult() is True
    assert child.is_adult() is False


def test_user_greeting() -> None:
    """Test string output."""
    user = User("Charlie", "charlie@test.com", 30)
    assert user.greeting() == "Hello, Charlie!"
    assert "Charlie" in user.greeting()


def test_divide() -> None:
    """Test function output."""
    assert divide(10, 2) == 5
    assert divide(7, 2) == 3.5
    assert divide(-6, 2) == -3


# =====================
# Testing Exceptions
# =====================

def test_divide_by_zero() -> None:
    """Test that exception is raised."""
    with pytest.raises(ValueError) as exc_info:
        divide(10, 0)

    assert "Cannot divide by zero" in str(exc_info.value)


def test_divide_by_zero_simple() -> None:
    """Simpler exception check."""
    with pytest.raises(ValueError, match="divide by zero"):
        divide(10, 0)


# =====================
# Fixtures
# =====================

@pytest.fixture
def sample_user() -> User:
    """Fixture providing a test user."""
    return User("Test User", "test@example.com", 25)


@pytest.fixture
def temp_config_file() -> str:
    """Fixture with cleanup."""
    # Setup
    fd, path = tempfile.mkstemp(suffix=".cfg")
    with os.fdopen(fd, "w") as f:
        f.write("host=localhost\n")
        f.write("port=8080\n")

    yield path  # Test runs here

    # Teardown
    os.unlink(path)


def test_with_fixture(sample_user: User) -> None:
    """Test using fixture."""
    assert sample_user.name == "Test User"
    assert sample_user.is_adult()


def test_read_config(temp_config_file: str) -> None:
    """Test with file fixture."""
    config = read_config(temp_config_file)
    assert config["host"] == "localhost"
    assert config["port"] == "8080"


# =====================
# Parametrized Tests
# =====================

@pytest.mark.parametrize("a,b,expected", [
    (10, 2, 5),
    (9, 3, 3),
    (7, 2, 3.5),
    (-6, 2, -3),
    (0, 5, 0),
])
def test_divide_parametrized(a: float, b: float, expected: float) -> None:
    """Test multiple cases with parametrize."""
    assert divide(a, b) == expected


@pytest.mark.parametrize("age,expected", [
    (17, False),
    (18, True),
    (19, True),
    (0, False),
    (100, True),
])
def test_is_adult_parametrized(age: int, expected: bool) -> None:
    """Test boundary conditions."""
    user = User("Test", "test@test.com", age)
    assert user.is_adult() == expected


# =====================
# Mocking
# =====================

def test_user_service_get_user() -> None:
    """Mock database dependency."""
    # Create mock database
    mock_db = Mock()
    mock_db.fetch.return_value = {
        "name": "Alice",
        "email": "alice@test.com",
        "age": 30
    }

    service = UserService(mock_db)
    user = service.get_user(1)

    assert user is not None
    assert user.name == "Alice"
    mock_db.fetch.assert_called_once_with(1)


def test_user_service_user_not_found() -> None:
    """Mock returning None."""
    mock_db = Mock()
    mock_db.fetch.return_value = None

    service = UserService(mock_db)
    user = service.get_user(999)

    assert user is None


def test_user_service_create_user() -> None:
    """Verify mock was called correctly."""
    mock_db = Mock()

    service = UserService(mock_db)
    user = service.create_user("Bob", "bob@test.com", 25)

    assert user.name == "Bob"
    mock_db.save.assert_called_once()
    saved_user = mock_db.save.call_args[0][0]
    assert saved_user.name == "Bob"


# =====================
# Patching
# =====================

def get_current_time() -> str:
    import datetime
    return datetime.datetime.now().isoformat()


def test_patch_datetime() -> None:
    """Patch module-level import."""
    with patch("datetime.datetime") as mock_datetime:
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T00:00:00"

        # This would use patched datetime
        # result = get_current_time()
        # assert result == "2024-01-01T00:00:00"
        pass  # Simplified for demo


@patch("os.path.exists")
def test_patch_decorator(mock_exists: Mock) -> None:
    """Patch using decorator."""
    mock_exists.return_value = True
    assert os.path.exists("/fake/path") is True
    mock_exists.assert_called_with("/fake/path")


# =====================
# Markers
# =====================

@pytest.mark.slow
def test_slow_operation() -> None:
    """Marked as slow test."""
    # Run with: pytest -m slow
    # Skip with: pytest -m "not slow"
    import time
    time.sleep(0.1)
    assert True


@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature() -> None:
    """Skipped test."""
    assert False


@pytest.mark.skipif(
    os.environ.get("CI") is None,
    reason="Only run in CI"
)
def test_ci_only() -> None:
    """Conditional skip."""
    assert True


@pytest.mark.xfail(reason="Known bug")
def test_known_bug() -> None:
    """Expected to fail."""
    assert 1 + 1 == 3


# =====================
# Fixture Scopes
# =====================

@pytest.fixture(scope="module")
def expensive_resource() -> str:
    """Fixture created once per module."""
    print("\nCreating expensive resource...")
    return "expensive_data"


@pytest.fixture(scope="session")
def database_connection() -> str:
    """Fixture created once per test session."""
    print("\nEstablishing database connection...")
    return "db_connection"


# =====================
# Conftest.py (shared fixtures)
# =====================

# In conftest.py, you can define fixtures available to all tests:
#
# @pytest.fixture
# def app():
#     return create_app(testing=True)
#
# @pytest.fixture
# def client(app):
#     return app.test_client()


# =====================
# Async Tests
# =====================

@pytest.mark.asyncio
async def test_async_function() -> None:
    """Test async function (requires pytest-asyncio)."""
    import asyncio

    async def async_add(a: int, b: int) -> int:
        await asyncio.sleep(0.01)
        return a + b

    result = await async_add(2, 3)
    assert result == 5


# =====================
# Test Organization Summary
# =====================

"""
Project structure:
├── src/
│   └── mypackage/
│       ├── __init__.py
│       └── module.py
├── tests/
│   ├── conftest.py      # Shared fixtures
│   ├── test_module.py   # Unit tests
│   └── integration/
│       └── test_api.py  # Integration tests
└── pytest.ini           # pytest configuration

pytest.ini example:
[pytest]
testpaths = tests
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
addopts = -v --tb=short

Common commands:
  pytest                     # Run all tests
  pytest tests/test_foo.py   # Run specific file
  pytest -k "test_user"      # Run tests matching pattern
  pytest -m slow             # Run marked tests
  pytest -x                  # Stop on first failure
  pytest --pdb               # Drop to debugger on failure
  pytest -v                  # Verbose output
  pytest --cov=src           # With coverage (pytest-cov)
"""


# =====================
# Run Demo
# =====================

if __name__ == "__main__":
    # Run with: pytest pytest_patterns.py -v
    print("Run with: pytest pytest_patterns.py -v")
    print("Or: python -m pytest pytest_patterns.py -v")
