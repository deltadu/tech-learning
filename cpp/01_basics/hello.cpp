/*
 * TOPIC: Hello World & Basic Syntax
 *
 * USE CASES:
 *   - First program to verify compiler/toolchain setup
 *   - Basic I/O with std::cout and std::cin
 *   - Simple loops and string handling
 *
 * KEY POINTS:
 *   - #include brings in library headers
 *   - std:: prefix for standard library (cout, cin, endl, string)
 *   - main() is the program entry point, returns int
 *   - Use std::getline() for input with spaces
 */

#include <iostream>
#include <string>

int main() {
    std::string name;
    
    std::cout << "Welcome to C++!" << std::endl;
    std::cout << "What's your name? ";
    std::getline(std::cin, name);
    
    std::cout << "\nHello, " << name << "! 👋" << std::endl;
    std::cout << "Here's a quick calculation: 7 * 6 = " << 7 * 6 << std::endl;
    
    // Simple loop demonstration
    std::cout << "\nCounting down: ";
    for (int i = 5; i > 0; --i) {
        std::cout << i << " ";
    }
    std::cout << "Liftoff! 🚀" << std::endl;
    
    return 0;
}
