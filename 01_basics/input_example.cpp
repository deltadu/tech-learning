#include <iostream>
#include <string>

int main() {
    // >> extracts data FROM input stream INTO variables
    
    int age;
    double height;
    std::string name;
    
    // Example 1: Read a single integer
    std::cout << "Enter your age: ";
    std::cin >> age;
    
    // Example 2: Read a double
    std::cout << "Enter your height (meters): ";
    std::cin >> height;
    
    // Example 3: Chain multiple >> operators
    int a, b, c;
    std::cout << "Enter 3 numbers separated by spaces: ";
    std::cin >> a >> b >> c;  // Reads all 3 at once!
    
    // Clear any leftover newline before getline
    std::cin.ignore();
    
    // Example 4: Read a full line (use getline instead of >> for strings with spaces)
    std::cout << "Enter your full name: ";
    std::getline(std::cin, name);
    
    // Summary
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Name: " << name << std::endl;
    std::cout << "Age: " << age << std::endl;
    std::cout << "Height: " << height << " meters" << std::endl;
    std::cout << "Numbers: " << a << ", " << b << ", " << c << std::endl;
    std::cout << "Sum: " << (a + b + c) << std::endl;
    
    return 0;
}
