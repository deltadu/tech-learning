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
