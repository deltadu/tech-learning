#include <iostream>

// Function declaration (prototype) - tells compiler the function exists
int add(int a, int b);
void greet(const std::string& name);

// Main function - program entry point
int main() {
    // Invoking (calling) functions:
    
    // 1. Call a function that returns a value
    int result = add(5, 3);
    std::cout << "5 + 3 = " << result << std::endl;
    
    // 2. Call a function directly in an expression
    std::cout << "10 + 20 = " << add(10, 20) << std::endl;
    
    // 3. Call a void function (no return value)
    greet("Alice");
    greet("Bob");
    
    return 0;
}

// Function definitions (implementations)
int add(int a, int b) {
    return a + b;
}

void greet(const std::string& name) {
    std::cout << "Hello, " << name << "!" << std::endl;
}
