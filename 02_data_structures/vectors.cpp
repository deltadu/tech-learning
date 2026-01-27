/*
 * TOPIC: Vectors (Dynamic Arrays)
 *
 * USE CASES:
 *   - Robotics: Store sensor readings, waypoints, trajectories
 *   - AV: Point clouds (LiDAR), detected objects list
 *   - Any collection that grows/shrinks at runtime
 *
 * KEY POINTS:
 *   - std::vector is THE default container in C++
 *   - Dynamic size (unlike fixed arrays)
 *   - push_back() adds, pop_back() removes from end
 *   - Access with [] or .at() (bounds-checked)
 *   - 2D vectors for matrices/grids
 */

#include <iostream>
#include <vector>

int main() {
    // =====================
    // 1D Vector (List)
    // =====================
    std::vector<int> numbers = {10, 20, 30};
    
    // Add elements
    numbers.push_back(40);
    numbers.push_back(50);
    
    // Access by index
    std::cout << "First element: " << numbers[0] << std::endl;
    std::cout << "Last element: " << numbers.back() << std::endl;
    
    // Modify element
    numbers[1] = 25;
    
    // Size
    std::cout << "Size: " << numbers.size() << std::endl;
    
    // Iterate with range-based for loop
    std::cout << "All elements: ";
    for (int n : numbers) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    // Remove last element
    numbers.pop_back();
    
    // Insert at position (insert 15 at index 1)
    numbers.insert(numbers.begin() + 1, 15);
    
    // Erase at position (remove element at index 2)
    numbers.erase(numbers.begin() + 2);
    
    std::cout << "After modifications: ";
    for (int n : numbers) {
        std::cout << n << " ";
    }
    std::cout << "\n\n";

    // =====================
    // 2D Vector (2D List / Matrix)
    // =====================
    std::vector<std::vector<int>> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    
    // Access element at row 1, col 2
    std::cout << "matrix[1][2] = " << matrix[1][2] << std::endl;
    
    // Modify element
    matrix[0][0] = 100;
    
    // Add a new row
    matrix.push_back({10, 11, 12});
    
    // Add element to existing row
    matrix[0].push_back(999);
    
    // Get dimensions
    std::cout << "Rows: " << matrix.size() << std::endl;
    std::cout << "Cols in row 0: " << matrix[0].size() << std::endl;
    
    // Print entire matrix
    std::cout << "\nMatrix contents:" << std::endl;
    for (const auto& row : matrix) {
        for (int val : row) {
            std::cout << val << "\t";
        }
        std::cout << std::endl;
    }
    
    return 0;
}
