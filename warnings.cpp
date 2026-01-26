#include <iostream>

int main() {
    int unused_variable = 42;  // Never used - Wall will warn!
    int x;                     // Uninitialized
    
    std::cout << x << std::endl;  // Using uninitialized value!
    
    return 0;
}
