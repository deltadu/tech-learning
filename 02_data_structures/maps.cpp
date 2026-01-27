/*
 * TOPIC: Maps (Dictionaries / Hash Tables)
 *
 * USE CASES:
 *   - Robotics: Parameter lookup, sensor ID -> readings
 *   - AV: Object tracking (ID -> object state)
 *   - Caching, configuration, any key-value storage
 *
 * KEY POINTS:
 *   - std::map = ordered by key (red-black tree), O(log n)
 *   - std::unordered_map = hash table, O(1) average
 *   - Use unordered_map for speed, map if order matters
 *   - Check existence with .find() or .count()
 *   - [] creates entry if missing; .at() throws if missing
 */

#include <iostream>
#include <map>
#include <unordered_map>
#include <string>

int main() {
    // =====================
    // std::map (ordered dictionary, sorted by key)
    // =====================
    std::map<std::string, int> ages;
    
    // Insert key-value pairs
    ages["Alice"] = 25;
    ages["Bob"] = 30;
    ages.insert({"Charlie", 35});
    
    // Access value by key
    std::cout << "Alice's age: " << ages["Alice"] << std::endl;
    
    // Modify value
    ages["Alice"] = 26;
    
    // Check if key exists
    if (ages.find("Bob") != ages.end()) {
        std::cout << "Bob exists in the map" << std::endl;
    }
    
    // Using count() to check existence (returns 0 or 1)
    if (ages.count("David") == 0) {
        std::cout << "David not found" << std::endl;
    }
    
    // Size
    std::cout << "Number of entries: " << ages.size() << std::endl;
    
    // Iterate through map (sorted by key)
    std::cout << "\nAll entries (sorted):" << std::endl;
    for (const auto& [name, age] : ages) {
        std::cout << "  " << name << " -> " << age << std::endl;
    }
    
    // Erase by key
    ages.erase("Charlie");
    
    std::cout << "\nAfter erasing Charlie:" << std::endl;
    for (const auto& pair : ages) {
        std::cout << "  " << pair.first << " -> " << pair.second << std::endl;
    }

    std::cout << "\n";

    // =====================
    // std::unordered_map (hash map, faster but unordered)
    // =====================
    std::unordered_map<std::string, std::string> capitals;
    
    capitals["USA"] = "Washington D.C.";
    capitals["France"] = "Paris";
    capitals["Japan"] = "Tokyo";
    capitals["Brazil"] = "Brasilia";
    
    std::cout << "Capital of Japan: " << capitals["Japan"] << std::endl;
    
    // Safe access with .at() - throws exception if key missing
    try {
        std::cout << "Capital of France: " << capitals.at("France") << std::endl;
    } catch (const std::out_of_range& e) {
        std::cout << "Key not found!" << std::endl;
    }
    
    // Iterate (order not guaranteed)
    std::cout << "\nAll capitals:" << std::endl;
    for (const auto& [country, capital] : capitals) {
        std::cout << "  " << country << " -> " << capital << std::endl;
    }
    
    return 0;
}
