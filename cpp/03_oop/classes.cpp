/*
 * TOPIC: Classes & Object-Oriented Programming
 *
 * USE CASES:
 *   - Robotics: Model sensors, actuators, robots as objects
 *   - AV: Vehicle, Obstacle, Planner classes
 *   - Any complex system with encapsulated state/behavior
 *
 * KEY POINTS:
 *   - private = internal state, public = interface
 *   - Constructors initialize objects (use initializer lists)
 *   - virtual + override = polymorphism (base ptr to derived)
 *   - static members shared across all instances
 *   - Operator overloading for intuitive syntax (+, ==, <<)
 */

#include <iostream>
#include <string>

// =====================
// Basic Class Definition
// =====================
class Dog {
private:
    // Member variables (attributes)
    std::string name;
    int age;

public:
    // Default constructor
    Dog() : name("Unknown"), age(0) {}

    // Parameterized constructor
    Dog(const std::string& name, int age) : name(name), age(age) {}

    // Getter methods
    std::string getName() const { return name; }
    int getAge() const { return age; }

    // Setter methods
    void setName(const std::string& newName) { name = newName; }
    void setAge(int newAge) { if (newAge >= 0) age = newAge; }

    // Member function
    void bark() const {
        std::cout << name << " says: Woof!" << std::endl;
    }

    // Display info
    void info() const {
        std::cout << "Dog: " << name << ", Age: " << age << std::endl;
    }
};

// =====================
// Static Members
// =====================
class Counter {
private:
    static int count;  // Shared across all instances
    int id;

public:
    Counter() : id(++count) {}

    int getId() const { return id; }
    static int getCount() { return count; }
};

// Static member must be defined outside the class
int Counter::count = 0;

// =====================
// Inheritance
// =====================
class Animal {
protected:
    std::string species;

public:
    Animal(const std::string& species) : species(species) {}

    virtual void speak() const {
        std::cout << "Some generic animal sound" << std::endl;
    }

    std::string getSpecies() const { return species; }

    virtual ~Animal() = default;
};

// Derived class
class Cat : public Animal {
private:
    std::string name;

public:
    Cat(const std::string& name) : Animal("Cat"), name(name) {}

    // Override base class method
    void speak() const override {
        std::cout << name << " says: Meow!" << std::endl;
    }

    std::string getName() const { return name; }
};

// =====================
// Operator Overloading
// =====================
class Point {
public:
    double x, y;

    Point(double x = 0, double y = 0) : x(x), y(y) {}

    // Overload + operator
    Point operator+(const Point& other) const {
        return Point(x + other.x, y + other.y);
    }

    // Overload == operator
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }

    void print() const {
        std::cout << "(" << x << ", " << y << ")" << std::endl;
    }
};

int main() {
    // =====================
    // Using Basic Class
    // =====================
    std::cout << "=== Basic Class ===" << std::endl;

    Dog dog1;                       // Default constructor
    Dog dog2("Buddy", 3);          // Parameterized constructor

    dog1.info();                   // "Dog: Unknown, Age: 0"
    dog2.info();                   // "Dog: Buddy, Age: 3"
    dog2.bark();                   // "Buddy says: Woof!"

    dog1.setName("Max");
    dog1.setAge(5);
    dog1.info();                   // "Dog: Max, Age: 5"

    std::cout << std::endl;

    // =====================
    // Using Static Members
    // =====================
    std::cout << "=== Static Members ===" << std::endl;

    Counter c1, c2, c3;
    std::cout << "c1 id: " << c1.getId() << std::endl;  // 1
    std::cout << "c2 id: " << c2.getId() << std::endl;  // 2
    std::cout << "c3 id: " << c3.getId() << std::endl;  // 3
    std::cout << "Total created: " << Counter::getCount() << std::endl;  // 3

    std::cout << std::endl;

    // =====================
    // Using Inheritance
    // =====================
    std::cout << "=== Inheritance ===" << std::endl;

    Animal* animal = new Animal("Generic");
    Cat* cat = new Cat("Whiskers");

    animal->speak();               // "Some generic animal sound"
    cat->speak();                  // "Whiskers says: Meow!"

    // Polymorphism: base pointer to derived object
    Animal* polyAnimal = new Cat("Luna");
    polyAnimal->speak();           // "Luna says: Meow!" (virtual dispatch)

    delete animal;
    delete cat;
    delete polyAnimal;

    std::cout << std::endl;

    // =====================
    // Using Operator Overloading
    // =====================
    std::cout << "=== Operator Overloading ===" << std::endl;

    Point p1(3, 4);
    Point p2(1, 2);
    Point p3 = p1 + p2;            // Uses overloaded +

    std::cout << "p1: "; p1.print();           // (3, 4)
    std::cout << "p2: "; p2.print();           // (1, 2)
    std::cout << "p1 + p2: "; p3.print();      // (4, 6)

    std::cout << "p1 == p2: " << (p1 == p2 ? "true" : "false") << std::endl;  // false
    std::cout << "p1 == p1: " << (p1 == p1 ? "true" : "false") << std::endl;  // true

    return 0;
}
