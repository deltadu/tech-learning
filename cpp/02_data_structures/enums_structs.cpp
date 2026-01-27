/*
 * TOPIC: Enums & Structs
 *
 * USE CASES:
 *   - Robotics: State machines (IDLE, MOVING, ERROR)
 *   - AV: Object types (CAR, PEDESTRIAN, CYCLIST)
 *   - Grouping related data (Point, Pose, SensorReading)
 *
 * KEY POINTS:
 *   - enum class = strongly typed, scoped (prefer over old enum)
 *   - struct = public by default, use for plain data
 *   - class = private by default, use for encapsulated objects
 *   - Structs can have methods, constructors, etc.
 *   - Use aggregate initialization: Point p = {1.0, 2.0}
 */

#include <iostream>
#include <string>
#include <vector>

// =====================
// Old-style enum (avoid - pollutes namespace)
// =====================
enum OldColor { RED, GREEN, BLUE };  // RED, GREEN, BLUE are global!

// =====================
// Enum Class (strongly typed - preferred!)
// =====================
enum class Color { Red, Green, Blue, Yellow };

enum class Size { Small, Medium, Large };

// Enum with explicit values
enum class HttpStatus {
    OK = 200,
    Created = 201,
    BadRequest = 400,
    NotFound = 404,
    InternalError = 500
};

// Enum with underlying type
enum class SmallEnum : uint8_t {
    A = 0,
    B = 1,
    C = 2
};

// =====================
// Basic Struct
// =====================
struct Point {
    double x;
    double y;

    // Structs can have methods too!
    double distanceFromOrigin() const {
        return std::sqrt(x * x + y * y);
    }
};

// =====================
// Struct with Constructor
// =====================
struct Person {
    std::string name;
    int age;
    std::string city;

    // Default constructor
    Person() : name("Unknown"), age(0), city("Unknown") {}

    // Parameterized constructor
    Person(const std::string& n, int a, const std::string& c)
        : name(n), age(a), city(c) {}

    void print() const {
        std::cout << name << ", " << age << " years old, from " << city << std::endl;
    }
};

// =====================
// Struct vs Class
// =====================
// The ONLY difference: struct defaults to public, class defaults to private
struct PublicByDefault {
    int x;  // public by default
};

class PrivateByDefault {
    int x;  // private by default
public:
    int getX() const { return x; }
};

// Convention: Use struct for plain data, class for objects with behavior

// =====================
// Nested Struct
// =====================
struct Rectangle {
    Point topLeft;
    Point bottomRight;

    double width() const { return bottomRight.x - topLeft.x; }
    double height() const { return bottomRight.y - topLeft.y; }
    double area() const { return width() * height(); }
};

int main() {
    // =====================
    // Using Enum Class
    // =====================
    std::cout << "=== Enum Class ===" << std::endl;

    Color c = Color::Red;  // Must use scope!

    // Switch on enum
    switch (c) {
        case Color::Red:    std::cout << "Color is Red" << std::endl; break;
        case Color::Green:  std::cout << "Color is Green" << std::endl; break;
        case Color::Blue:   std::cout << "Color is Blue" << std::endl; break;
        case Color::Yellow: std::cout << "Color is Yellow" << std::endl; break;
    }

    // Compare enums
    if (c == Color::Red) {
        std::cout << "It's red!" << std::endl;
    }

    // Cannot mix different enum types (compile error)
    // if (c == Size::Small) {}  // ERROR!

    // Convert to int (explicit cast required)
    int colorValue = static_cast<int>(c);
    std::cout << "Red as int: " << colorValue << std::endl;

    // HttpStatus example
    HttpStatus status = HttpStatus::NotFound;
    std::cout << "HTTP status: " << static_cast<int>(status) << std::endl;

    std::cout << std::endl;

    // =====================
    // Using Structs
    // =====================
    std::cout << "=== Structs ===" << std::endl;

    // Aggregate initialization
    Point p1 = {3.0, 4.0};
    std::cout << "Point: (" << p1.x << ", " << p1.y << ")" << std::endl;
    std::cout << "Distance from origin: " << p1.distanceFromOrigin() << std::endl;

    // Designated initializers (C++20)
    // Point p2 = {.x = 5.0, .y = 12.0};

    // Access and modify
    p1.x = 6.0;
    p1.y = 8.0;
    std::cout << "Modified distance: " << p1.distanceFromOrigin() << std::endl;

    std::cout << std::endl;

    // =====================
    // Struct with Constructor
    // =====================
    std::cout << "=== Person Struct ===" << std::endl;

    Person person1;  // Default constructor
    Person person2("Alice", 25, "NYC");  // Parameterized

    person1.print();
    person2.print();

    // Modify fields directly (they're public)
    person1.name = "Bob";
    person1.age = 30;
    person1.city = "LA";
    person1.print();

    std::cout << std::endl;

    // =====================
    // Vector of Structs
    // =====================
    std::cout << "=== Vector of Structs ===" << std::endl;

    std::vector<Person> people = {
        {"Charlie", 35, "Chicago"},
        {"Diana", 28, "Boston"},
        {"Eve", 22, "Seattle"}
    };

    for (const auto& p : people) {
        p.print();
    }

    // Add more
    people.push_back({"Frank", 40, "Denver"});
    people.emplace_back("Grace", 33, "Miami");  // Construct in-place

    std::cout << "Total people: " << people.size() << std::endl;

    std::cout << std::endl;

    // =====================
    // Nested Structs
    // =====================
    std::cout << "=== Nested Structs ===" << std::endl;

    Rectangle rect = {{0, 0}, {10, 5}};
    std::cout << "Rectangle: (" << rect.topLeft.x << "," << rect.topLeft.y << ") to ("
              << rect.bottomRight.x << "," << rect.bottomRight.y << ")" << std::endl;
    std::cout << "Width: " << rect.width() << std::endl;
    std::cout << "Height: " << rect.height() << std::endl;
    std::cout << "Area: " << rect.area() << std::endl;

    std::cout << std::endl;

    // =====================
    // Struct Comparison
    // =====================
    std::cout << "=== Struct Comparison ===" << std::endl;

    Point a = {1.0, 2.0};
    Point b = {1.0, 2.0};

    // Structs don't have == by default, must define or compare fields
    bool equal = (a.x == b.x && a.y == b.y);
    std::cout << "a == b: " << equal << std::endl;

    Point c2 = {3.0, 4.0};
    std::cout << "a == c2: " << (a.x == c2.x && a.y == c2.y) << std::endl;

    // C++20 adds default comparison with <=> (spaceship operator)

    std::cout << std::endl;

    // =====================
    // Summary
    // =====================
    std::cout << "=== When to Use ===" << std::endl;
    std::cout << "enum class: Named constants, state machines, options" << std::endl;
    std::cout << "struct: Plain data grouping, simple types" << std::endl;
    std::cout << "class: Complex objects with encapsulation" << std::endl;

    return 0;
}
