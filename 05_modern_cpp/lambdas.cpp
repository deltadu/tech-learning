/*
 * TOPIC: Lambda Expressions
 *
 * USE CASES:
 *   - STL algorithms (sort, find_if, transform)
 *   - Callbacks, event handlers
 *   - ROS2: Timer callbacks, subscriber handlers
 *   - Short, one-off functions inline
 *
 * KEY POINTS:
 *   - [capture](params) { body }
 *   - [=] capture all by value, [&] by reference
 *   - [x, &y] mixed: x by value, y by reference
 *   - mutable allows modifying captured copies
 *   - auto for storage, std::function for type erasure
 *   - Generic lambdas use auto parameters (C++14)
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>  // std::function

int main() {
    // =====================
    // Basic Lambda Syntax
    // =====================
    std::cout << "=== Basic Lambda ===" << std::endl;

    // Simplest lambda
    auto sayHello = []() {
        std::cout << "Hello from lambda!" << std::endl;
    };
    sayHello();

    // Lambda with parameters
    auto add = [](int a, int b) {
        return a + b;
    };
    std::cout << "3 + 4 = " << add(3, 4) << std::endl;

    // Explicit return type (usually inferred)
    auto divide = [](double a, double b) -> double {
        return a / b;
    };
    std::cout << "10 / 3 = " << divide(10, 3) << std::endl;

    std::cout << std::endl;

    // =====================
    // Capture Clauses
    // =====================
    std::cout << "=== Capture Clauses ===" << std::endl;

    int x = 10;
    int y = 20;

    // [=] Capture all by VALUE (copy)
    auto captureByValue = [=]() {
        std::cout << "[=] x=" << x << ", y=" << y << std::endl;
        // x++;  // ERROR: captured by value is const by default
    };
    captureByValue();

    // [&] Capture all by REFERENCE
    auto captureByRef = [&]() {
        x++;
        y++;
        std::cout << "[&] modified x=" << x << ", y=" << y << std::endl;
    };
    captureByRef();
    std::cout << "After [&]: x=" << x << ", y=" << y << std::endl;

    // [x, &y] Mixed: x by value, y by reference
    auto mixed = [x, &y]() {
        // x is copied, y is referenced
        y += 100;
        std::cout << "[x, &y] x=" << x << ", y=" << y << std::endl;
    };
    mixed();

    // [this] Capture this pointer (in class methods)
    // [*this] Capture copy of *this (C++17)

    std::cout << std::endl;

    // =====================
    // Mutable Lambdas
    // =====================
    std::cout << "=== Mutable Lambda ===" << std::endl;

    int counter = 0;

    // Without mutable - cannot modify captured values
    // auto inc = [counter]() { counter++; };  // ERROR!

    // With mutable - can modify the COPY
    auto inc = [counter]() mutable {
        counter++;
        std::cout << "Inside lambda: " << counter << std::endl;
        return counter;
    };

    std::cout << "Call 1: " << inc() << std::endl;  // 1
    std::cout << "Call 2: " << inc() << std::endl;  // 2
    std::cout << "Original counter: " << counter << std::endl;  // 0 (unchanged!)

    std::cout << std::endl;

    // =====================
    // Lambdas with STL Algorithms
    // =====================
    std::cout << "=== Lambdas with STL ===" << std::endl;

    std::vector<int> nums = {5, 2, 8, 1, 9, 3, 7};

    // Sort with custom comparator
    std::sort(nums.begin(), nums.end(), [](int a, int b) {
        return a > b;  // Descending
    });
    std::cout << "Sorted desc: ";
    for (int n : nums) std::cout << n << " ";
    std::cout << std::endl;

    // find_if
    auto it = std::find_if(nums.begin(), nums.end(), [](int n) {
        return n < 5;
    });
    std::cout << "First < 5: " << *it << std::endl;

    // count_if
    int evens = std::count_if(nums.begin(), nums.end(), [](int n) {
        return n % 2 == 0;
    });
    std::cout << "Even count: " << evens << std::endl;

    // for_each
    std::cout << "Doubled: ";
    std::for_each(nums.begin(), nums.end(), [](int n) {
        std::cout << n * 2 << " ";
    });
    std::cout << std::endl;

    // transform with capture
    int multiplier = 3;
    std::vector<int> result(nums.size());
    std::transform(nums.begin(), nums.end(), result.begin(), [multiplier](int n) {
        return n * multiplier;
    });
    std::cout << "Tripled: ";
    for (int n : result) std::cout << n << " ";
    std::cout << std::endl;

    std::cout << std::endl;

    // =====================
    // Storing Lambdas
    // =====================
    std::cout << "=== Storing Lambdas ===" << std::endl;

    // auto (preferred, most efficient)
    auto lambda1 = [](int x) { return x * x; };

    // std::function (flexible but has overhead)
    std::function<int(int)> lambda2 = [](int x) { return x * x; };

    std::cout << "lambda1(5) = " << lambda1(5) << std::endl;
    std::cout << "lambda2(5) = " << lambda2(5) << std::endl;

    // Function pointer (only for stateless lambdas - no captures)
    int (*funcPtr)(int, int) = [](int a, int b) { return a + b; };
    std::cout << "funcPtr(3, 4) = " << funcPtr(3, 4) << std::endl;

    std::cout << std::endl;

    // =====================
    // Generic Lambdas (C++14)
    // =====================
    std::cout << "=== Generic Lambda (auto params) ===" << std::endl;

    auto genericAdd = [](auto a, auto b) {
        return a + b;
    };

    std::cout << "int: " << genericAdd(1, 2) << std::endl;
    std::cout << "double: " << genericAdd(1.5, 2.5) << std::endl;
    std::cout << "string: " << genericAdd(std::string("Hello "), std::string("World")) << std::endl;

    std::cout << std::endl;

    // =====================
    // Immediately Invoked Lambda (IIFE)
    // =====================
    std::cout << "=== Immediately Invoked ===" << std::endl;

    // Useful for complex initialization
    const int result2 = []() {
        int temp = 0;
        for (int i = 1; i <= 10; i++) {
            temp += i;
        }
        return temp;
    }();  // Note the () at the end!

    std::cout << "Sum 1-10: " << result2 << std::endl;

    // =====================
    // Summary
    // =====================
    std::cout << std::endl;
    std::cout << "=== Capture Summary ===" << std::endl;
    std::cout << "[]      - Capture nothing" << std::endl;
    std::cout << "[=]     - Capture all by value" << std::endl;
    std::cout << "[&]     - Capture all by reference" << std::endl;
    std::cout << "[x]     - Capture x by value" << std::endl;
    std::cout << "[&x]    - Capture x by reference" << std::endl;
    std::cout << "[=, &x] - All by value, x by reference" << std::endl;
    std::cout << "[&, x]  - All by reference, x by value" << std::endl;
    std::cout << "[this]  - Capture this pointer" << std::endl;

    return 0;
}
