# C++ Quick Reference

## Compiling & Running

```bash
# Basic compile
g++ -o program program.cpp

# With warnings (recommended)
g++ -Wall -o program program.cpp

# With C++17 features
g++ -Wall -std=c++17 -o program program.cpp

# Run
./program
```

---

## I/O Streams

| Object | Meaning | Operator |
|--------|---------|----------|
| `std::cout` | Output to screen | `<<` (insertion) |
| `std::cin` | Input from keyboard | `>>` (extraction) |
| `std::endl` | Newline + flush | |

```cpp
#include <iostream>

std::cout << "Hello " << name << std::endl;
std::cin >> number;
std::getline(std::cin, line);  // Read entire line with spaces
```

---

## Vector (Dynamic Array / List)

```cpp
#include <vector>
```

| Operation | Code |
|-----------|------|
| Create empty | `std::vector<int> v;` |
| Create with values | `std::vector<int> v = {1, 2, 3};` |
| Create with size | `std::vector<int> v(10);` (10 zeros) |
| Create with size & value | `std::vector<int> v(10, 5);` (10 fives) |
| Add to end | `v.push_back(4);` |
| Remove from end | `v.pop_back();` |
| Access by index | `v[0]` or `v.at(0)` |
| First element | `v.front()` |
| Last element | `v.back()` |
| Size | `v.size()` |
| Check empty | `v.empty()` |
| Clear all | `v.clear()` |
| Insert at index i | `v.insert(v.begin() + i, value);` |
| Erase at index i | `v.erase(v.begin() + i);` |

### Iterating

```cpp
// Range-based (preferred)
for (int x : v) {
    std::cout << x;
}

// With index
for (size_t i = 0; i < v.size(); ++i) {
    std::cout << v[i];
}

// By reference (to modify or avoid copy)
for (int& x : v) {
    x *= 2;
}
```

---

## 2D Vector (Matrix)

```cpp
#include <vector>
```

| Operation | Code |
|-----------|------|
| Create | `std::vector<std::vector<int>> m = {{1,2}, {3,4}};` |
| Create empty rows x cols | `std::vector<std::vector<int>> m(rows, std::vector<int>(cols));` |
| Create with default value | `std::vector<std::vector<int>> m(rows, std::vector<int>(cols, val));` |
| Access element | `m[row][col]` |
| Number of rows | `m.size()` |
| Number of cols in row i | `m[i].size()` |
| Add new row | `m.push_back({5, 6, 7});` |
| Add to existing row | `m[0].push_back(99);` |

### Iterating

```cpp
for (const auto& row : m) {
    for (int val : row) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}
```

---

## Map (Dictionary / Hash Table)

```cpp
#include <map>           // Sorted by key, O(log n) lookup
#include <unordered_map> // Unsorted, O(1) average lookup
```

| Operation | Code |
|-----------|------|
| Create | `std::map<std::string, int> m;` |
| Insert/Update | `m["key"] = value;` |
| Insert (alternative) | `m.insert({"key", value});` |
| Access | `m["key"]` |
| Safe access (throws if missing) | `m.at("key")` |
| Check if key exists | `m.count("key") > 0` |
| Check if key exists (alt) | `m.find("key") != m.end()` |
| Size | `m.size()` |
| Check empty | `m.empty()` |
| Erase by key | `m.erase("key");` |
| Clear all | `m.clear()` |

### Iterating

```cpp
// C++17 structured bindings (preferred)
for (const auto& [key, value] : m) {
    std::cout << key << " -> " << value;
}

// Pre-C++17
for (const auto& pair : m) {
    std::cout << pair.first << " -> " << pair.second;
}
```

### map vs unordered_map

| Type | Order | Lookup | Use When |
|------|-------|--------|----------|
| `std::map` | Sorted by key | O(log n) | Need sorted iteration |
| `std::unordered_map` | No order | O(1) avg | Need fast lookups |

---

## String Operations

```cpp
#include <string>
```

| Operation | Code |
|-----------|------|
| Create | `std::string s = "hello";` |
| Length | `s.length()` or `s.size()` |
| Access char | `s[0]` or `s.at(0)` |
| Concatenate | `s + " world"` or `s += " world"` |
| Substring | `s.substr(start, length)` |
| Find | `s.find("ell")` (returns index or `std::string::npos`) |
| Compare | `s == "hello"` or `s.compare("hello")` |
| Check empty | `s.empty()` |
| Clear | `s.clear()` |

---

## Functions

```cpp
// Declaration (prototype)
int add(int a, int b);
void greet(const std::string& name);

// Definition
int add(int a, int b) {
    return a + b;
}

void greet(const std::string& name) {
    std::cout << "Hello, " << name << std::endl;
}

// Invocation
int result = add(5, 3);
greet("Alice");
```

---

## Useful Compiler Flags

| Flag | Purpose |
|------|---------|
| `-Wall` | Enable common warnings |
| `-Wextra` | Enable extra warnings |
| `-std=c++17` | Use C++17 standard |
| `-std=c++20` | Use C++20 standard |
| `-O2` | Optimize for speed |
| `-g` | Include debug symbols |
