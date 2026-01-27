"""
TOPIC: Python Gotchas & Tips

USE CASES:
  - Avoiding common bugs
  - Writing more Pythonic code
  - Understanding tricky behavior

KEY POINTS:
  - Mutable default arguments trap
  - Late binding in closures
  - is vs == difference
  - Copy vs reference pitfalls
"""

from typing import Any
import copy

# =====================
# GOTCHA 1: Mutable Default Arguments
# =====================

def demo_mutable_default() -> None:
    print("=== Mutable Default Arguments ===\n")

    # BAD: Mutable default is shared across calls!
    def bad_append(item: int, lst: list[int] = []) -> list[int]:
        lst.append(item)
        return lst

    print(f"bad_append(1): {bad_append(1)}")  # [1]
    print(f"bad_append(2): {bad_append(2)}")  # [1, 2] - Oops!
    print(f"bad_append(3): {bad_append(3)}")  # [1, 2, 3] - Still accumulating!

    # GOOD: Use None as default
    def good_append(item: int, lst: list[int] | None = None) -> list[int]:
        if lst is None:
            lst = []
        lst.append(item)
        return lst

    print(f"\ngood_append(1): {good_append(1)}")  # [1]
    print(f"good_append(2): {good_append(2)}")  # [2] - Fresh list!
    print()


# =====================
# GOTCHA 2: Late Binding in Closures
# =====================

def demo_late_binding() -> None:
    print("=== Late Binding in Closures ===\n")

    # BAD: All lambdas capture the same variable
    funcs_bad = [lambda: i for i in range(3)]
    print(f"Bad: {[f() for f in funcs_bad]}")  # [2, 2, 2] - All 2!

    # GOOD: Capture value with default argument
    funcs_good = [lambda i=i: i for i in range(3)]
    print(f"Good: {[f() for f in funcs_good]}")  # [0, 1, 2]

    # Also works with partial
    from functools import partial
    funcs_partial = [partial(lambda x: x, i) for i in range(3)]
    print(f"Partial: {[f() for f in funcs_partial]}")  # [0, 1, 2]
    print()


# =====================
# GOTCHA 3: is vs ==
# =====================

def demo_is_vs_equal() -> None:
    print("=== is vs == ===\n")

    # == compares values, is compares identity

    a = [1, 2, 3]
    b = [1, 2, 3]
    c = a

    print(f"a = {a}, b = {b}")
    print(f"a == b: {a == b}")  # True (same value)
    print(f"a is b: {a is b}")  # False (different objects)
    print(f"a is c: {a is c}")  # True (same object)

    # Small integer caching (-5 to 256)
    x = 256
    y = 256
    print(f"\n256 is 256: {x is y}")  # True (cached)

    x = 257
    y = 257
    print(f"257 is 257: {x is y}")  # May be False!

    # Always use == for value comparison
    # Only use is for None, True, False
    print(f"\nCorrect: x is None, not x == None")
    print()


# =====================
# GOTCHA 4: Shallow vs Deep Copy
# =====================

def demo_copy() -> None:
    print("=== Shallow vs Deep Copy ===\n")

    original = [[1, 2], [3, 4]]

    # Assignment (same object)
    ref = original
    ref[0][0] = 99
    print(f"Assignment: original = {original}")  # [[99, 2], [3, 4]]

    original = [[1, 2], [3, 4]]  # Reset

    # Shallow copy (new list, same inner objects)
    shallow = original.copy()  # or list(original) or original[:]
    shallow[0][0] = 99
    print(f"Shallow copy: original = {original}")  # [[99, 2], [3, 4]] - Modified!

    original = [[1, 2], [3, 4]]  # Reset

    # Deep copy (completely independent)
    deep = copy.deepcopy(original)
    deep[0][0] = 99
    print(f"Deep copy: original = {original}")  # [[1, 2], [3, 4]] - Unchanged!
    print()


# =====================
# GOTCHA 5: Modifying List While Iterating
# =====================

def demo_modify_while_iterating() -> None:
    print("=== Modifying While Iterating ===\n")

    # BAD: Modifying list while iterating
    items = [1, 2, 3, 4, 5]
    # for item in items:
    #     if item % 2 == 0:
    #         items.remove(item)  # Skips elements!

    # GOOD: Create new list
    items = [1, 2, 3, 4, 5]
    items = [x for x in items if x % 2 != 0]
    print(f"Comprehension: {items}")  # [1, 3, 5]

    # GOOD: Iterate over copy
    items = [1, 2, 3, 4, 5]
    for item in items[:]:  # Slice creates copy
        if item % 2 == 0:
            items.remove(item)
    print(f"Iterate copy: {items}")  # [1, 3, 5]

    # GOOD: Iterate backwards for removal
    items = [1, 2, 3, 4, 5]
    for i in range(len(items) - 1, -1, -1):
        if items[i] % 2 == 0:
            del items[i]
    print(f"Backwards: {items}")  # [1, 3, 5]
    print()


# =====================
# GOTCHA 6: String Formatting Edge Cases
# =====================

def demo_string_formatting() -> None:
    print("=== String Formatting ===\n")

    # f-strings with dictionaries need different quotes
    d = {"key": "value"}
    print(f"Dict: {d['key']}")  # Works
    # print(f"Dict: {d["key"]}")  # SyntaxError!

    # Escaping braces
    print(f"Braces: {{not interpolated}}")

    # Debug with = (Python 3.8+)
    x = 42
    print(f"{x = }")  # x = 42

    # Format specs
    pi = 3.14159
    print(f"Pi: {pi:.2f}")  # 3.14
    print(f"Aligned: {pi:>10.2f}")  # Right-aligned
    print()


# =====================
# GOTCHA 7: Integer Division
# =====================

def demo_division() -> None:
    print("=== Integer Division ===\n")

    # / always returns float, // returns int
    print(f"5 / 2 = {5 / 2}")    # 2.5
    print(f"5 // 2 = {5 // 2}")  # 2

    # Negative division rounds toward negative infinity
    print(f"-5 // 2 = {-5 // 2}")   # -3 (not -2!)
    print(f"int(-5 / 2) = {int(-5 / 2)}")  # -2 (truncates toward zero)

    # For truncation toward zero, use int()
    print()


# =====================
# GOTCHA 8: Boolean Evaluation
# =====================

def demo_boolean() -> None:
    print("=== Boolean Evaluation ===\n")

    # Falsy values
    falsy = [False, None, 0, 0.0, "", [], {}, set()]
    for val in falsy:
        print(f"bool({val!r:10}) = {bool(val)}")

    print()

    # and/or return values, not True/False
    print(f"'a' or 'b' = {'a' or 'b'}")  # 'a'
    print(f"'' or 'b' = {'' or 'b'}")    # 'b'
    print(f"'a' and 'b' = {'a' and 'b'}")  # 'b'
    print(f"'' and 'b' = {'' and 'b'}")  # ''
    print()


# =====================
# TIP 1: Walrus Operator :=
# =====================

def demo_walrus() -> None:
    print("=== Walrus Operator := (Python 3.8+) ===\n")

    # Assign and use in one expression
    data = [1, 2, 3, 4, 5]

    # Without walrus
    n = len(data)
    if n > 3:
        print(f"Length {n} is > 3")

    # With walrus
    if (n := len(data)) > 3:
        print(f"With walrus: Length {n} is > 3")

    # Useful in while loops
    # while (line := file.readline()):
    #     process(line)

    # In list comprehensions
    results = [y for x in data if (y := x * 2) > 5]
    print(f"Filtered doubles: {results}")
    print()


# =====================
# TIP 2: Unpacking
# =====================

def demo_unpacking() -> None:
    print("=== Unpacking ===\n")

    # Extended unpacking
    first, *middle, last = [1, 2, 3, 4, 5]
    print(f"first={first}, middle={middle}, last={last}")

    # Ignore values
    first, *_, last = [1, 2, 3, 4, 5]
    print(f"first={first}, last={last}")

    # Swap without temp
    a, b = 1, 2
    a, b = b, a
    print(f"Swapped: a={a}, b={b}")

    # Unpack in function calls
    args = [1, 2, 3]
    kwargs = {"sep": "-", "end": "!\n"}
    print(*args, **kwargs)  # 1-2-3!
    print()


# =====================
# TIP 3: Dict Tricks
# =====================

def demo_dict_tricks() -> None:
    print("=== Dict Tricks ===\n")

    # setdefault
    d: dict[str, list[int]] = {}
    d.setdefault("key", []).append(1)
    d.setdefault("key", []).append(2)
    print(f"setdefault: {d}")

    # defaultdict is even better
    from collections import defaultdict
    dd: defaultdict[str, list[int]] = defaultdict(list)
    dd["key"].append(1)
    dd["key"].append(2)
    print(f"defaultdict: {dict(dd)}")

    # Merge dicts (Python 3.9+)
    a = {"x": 1, "y": 2}
    b = {"y": 3, "z": 4}
    merged = a | b  # b overwrites a
    print(f"Merged: {merged}")

    # Dict comprehension with enumerate
    items = ["a", "b", "c"]
    indexed = {v: i for i, v in enumerate(items)}
    print(f"Indexed: {indexed}")
    print()


# =====================
# TIP 4: Chained Comparisons
# =====================

def demo_chained_comparisons() -> None:
    print("=== Chained Comparisons ===\n")

    x = 5

    # Instead of: x > 0 and x < 10
    if 0 < x < 10:
        print(f"{x} is between 0 and 10")

    # Works with any comparison
    a, b, c = 1, 2, 3
    if a < b < c:
        print(f"{a} < {b} < {c}")

    # Even with equality
    if a <= b <= c:
        print(f"{a} <= {b} <= {c}")
    print()


# =====================
# TIP 5: else with for/while
# =====================

def demo_for_else() -> None:
    print("=== for/else ===\n")

    # else runs if loop completes without break
    items = [1, 3, 5, 7, 9]

    for item in items:
        if item == 6:
            print("Found 6!")
            break
    else:
        print("6 not found")  # This runs

    # Useful for search patterns
    primes = [2, 3, 5, 7, 11]
    n = 11
    for prime in primes:
        if n == prime:
            print(f"{n} is prime")
            break
    else:
        print(f"{n} is not in prime list")
    print()


# =====================
# Demo
# =====================

if __name__ == "__main__":
    print("=== Python Gotchas & Tips ===\n")

    demo_mutable_default()
    demo_late_binding()
    demo_is_vs_equal()
    demo_copy()
    demo_modify_while_iterating()
    demo_string_formatting()
    demo_division()
    demo_boolean()
    demo_walrus()
    demo_unpacking()
    demo_dict_tricks()
    demo_chained_comparisons()
    demo_for_else()
