#include <iostream>
#include <vector>

// =====================
// Pass by Value vs Reference vs Pointer
// =====================

// Pass by value - makes a COPY (original unchanged)
void incrementByValue(int x) {
    x++;  // Only modifies local copy
}

// Pass by reference - modifies ORIGINAL
void incrementByRef(int& x) {
    x++;  // Modifies caller's variable
}

// Pass by pointer - modifies ORIGINAL via address
void incrementByPtr(int* x) {
    (*x)++;  // Dereference to access value
}

// Const reference - read-only, no copy (efficient for large objects)
void printVector(const std::vector<int>& v) {
    for (int n : v) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
}

int main() {
    // =====================
    // Basic Pointers
    // =====================
    std::cout << "=== Basic Pointers ===" << std::endl;

    int value = 42;
    int* ptr = &value;  // ptr stores ADDRESS of value

    std::cout << "value: " << value << std::endl;        // 42
    std::cout << "&value (address): " << &value << std::endl;  // 0x7fff...
    std::cout << "ptr (address): " << ptr << std::endl;        // Same address
    std::cout << "*ptr (dereferenced): " << *ptr << std::endl; // 42

    // Modify through pointer
    *ptr = 100;
    std::cout << "After *ptr = 100, value: " << value << std::endl;  // 100

    std::cout << std::endl;

    // =====================
    // Null Pointers
    // =====================
    std::cout << "=== Null Pointers ===" << std::endl;

    int* nullPtr = nullptr;  // Points to nothing

    // Always check before dereferencing!
    if (nullPtr != nullptr) {
        std::cout << *nullPtr << std::endl;
    } else {
        std::cout << "Pointer is null, cannot dereference" << std::endl;
    }

    std::cout << std::endl;

    // =====================
    // References
    // =====================
    std::cout << "=== References ===" << std::endl;

    int original = 50;
    int& ref = original;  // ref IS original (alias)

    std::cout << "original: " << original << std::endl;  // 50
    std::cout << "ref: " << ref << std::endl;            // 50

    ref = 75;
    std::cout << "After ref = 75, original: " << original << std::endl;  // 75

    // Key difference: references CANNOT be null or reassigned
    int another = 999;
    ref = another;  // This COPIES value, doesn't reassign reference!
    std::cout << "After ref = another:" << std::endl;
    std::cout << "  original: " << original << std::endl;  // 999 (copied!)
    std::cout << "  another: " << another << std::endl;    // 999

    std::cout << std::endl;

    // =====================
    // Pass by Value vs Reference vs Pointer
    // =====================
    std::cout << "=== Function Parameters ===" << std::endl;

    int num = 10;
    std::cout << "Initial: " << num << std::endl;  // 10

    incrementByValue(num);
    std::cout << "After incrementByValue: " << num << std::endl;  // 10 (unchanged!)

    incrementByRef(num);
    std::cout << "After incrementByRef: " << num << std::endl;  // 11

    incrementByPtr(&num);
    std::cout << "After incrementByPtr: " << num << std::endl;  // 12

    std::cout << std::endl;

    // =====================
    // Pointer Arithmetic (Arrays)
    // =====================
    std::cout << "=== Pointer Arithmetic ===" << std::endl;

    int arr[] = {10, 20, 30, 40, 50};
    int* p = arr;  // Points to first element

    std::cout << "*p: " << *p << std::endl;        // 10
    std::cout << "*(p+1): " << *(p + 1) << std::endl;  // 20
    std::cout << "*(p+2): " << *(p + 2) << std::endl;  // 30

    p++;  // Move to next element
    std::cout << "After p++, *p: " << *p << std::endl;  // 20

    std::cout << std::endl;

    // =====================
    // When to Use What
    // =====================
    std::cout << "=== Guidelines ===" << std::endl;
    std::cout << "Use REFERENCE when:" << std::endl;
    std::cout << "  - You want to modify the original" << std::endl;
    std::cout << "  - Passing large objects (avoid copy)" << std::endl;
    std::cout << "  - Value is never null" << std::endl;
    std::cout << std::endl;
    std::cout << "Use POINTER when:" << std::endl;
    std::cout << "  - Value might be null/optional" << std::endl;
    std::cout << "  - Need to reassign to different object" << std::endl;
    std::cout << "  - Dynamic memory allocation" << std::endl;
    std::cout << "  - Polymorphism (base* to derived)" << std::endl;
    std::cout << std::endl;
    std::cout << "Use CONST REFERENCE for read-only large objects:" << std::endl;
    std::cout << "  void process(const std::vector<int>& data)" << std::endl;

    return 0;
}
