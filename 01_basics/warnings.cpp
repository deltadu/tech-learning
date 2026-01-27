/*
 * TOPIC: Compiler Warnings
 *
 * USE CASES:
 *   - Catch bugs before runtime
 *   - Safety-critical code (robotics, AV) requires warning-free builds
 *   - CI/CD pipelines often treat warnings as errors
 *
 * KEY POINTS:
 *   - Compile with -Wall -Wextra to enable warnings
 *   - Fix warnings, don't ignore them
 *   - Uninitialized variables = undefined behavior
 *   - Unused variables may indicate logic errors
 */

#include <iostream>

int main() {
    int unused_variable = 42;  // Never used - Wall will warn!
    int x;                     // Uninitialized
    
    std::cout << x << std::endl;  // Using uninitialized value!
    
    return 0;
}
