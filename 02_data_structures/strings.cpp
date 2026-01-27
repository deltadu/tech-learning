/*
 * TOPIC: Strings
 *
 * USE CASES:
 *   - Logging, debugging messages
 *   - Parsing config files, sensor data formats
 *   - Building commands for hardware communication
 *
 * KEY POINTS:
 *   - std::string is mutable (unlike Python)
 *   - Concatenate with + or +=
 *   - .find() returns npos if not found
 *   - Use std::stoi/stod for string->number
 *   - Use std::to_string for number->string
 *   - Splitting requires stringstream (no built-in split)
 */

#include <iostream>
#include <string>
#include <algorithm>  // for transform, reverse
#include <sstream>    // for stringstream (splitting)
#include <vector>

int main() {
    // =====================
    // Basic Operations
    // =====================
    std::string s = "Hello";

    // Concatenation
    s += " World";                    // Append
    std::string greeting = s + "!";   // Combine
    std::cout << greeting << std::endl;  // "Hello World!"

    // Length
    std::cout << "Length: " << s.length() << std::endl;  // 11

    // Access characters
    std::cout << "First char: " << s[0] << std::endl;      // 'H'
    std::cout << "Last char: " << s.back() << std::endl;   // 'd'

    // Modify character
    s[0] = 'h';
    std::cout << "Modified: " << s << std::endl;  // "hello World"

    std::cout << "\n";

    // =====================
    // Searching
    // =====================
    std::string text = "The quick brown fox jumps over the lazy dog";

    // Find substring (returns index, or std::string::npos if not found)
    size_t pos = text.find("fox");
    if (pos != std::string::npos) {
        std::cout << "'fox' found at index: " << pos << std::endl;  // 16
    }

    // Find from end
    pos = text.rfind("the");
    std::cout << "Last 'the' at index: " << pos << std::endl;  // 31

    // Check if contains (C++23 has .contains(), but this works everywhere)
    bool has_dog = text.find("dog") != std::string::npos;
    std::cout << "Contains 'dog': " << (has_dog ? "yes" : "no") << std::endl;

    // Find any character from a set
    pos = text.find_first_of("aeiou");
    std::cout << "First vowel at: " << pos << std::endl;  // 2 ('e')

    std::cout << "\n";

    // =====================
    // Substring & Replace
    // =====================
    std::string original = "Hello World";

    // Extract substring: substr(start, length)
    std::string sub = original.substr(0, 5);
    std::cout << "Substring: " << sub << std::endl;  // "Hello"

    // From position to end
    std::string rest = original.substr(6);
    std::cout << "Rest: " << rest << std::endl;  // "World"

    // Replace: replace(start, length, new_string)
    std::string modified = original;
    modified.replace(6, 5, "C++");
    std::cout << "Replaced: " << modified << std::endl;  // "Hello C++"

    // Erase characters
    std::string erased = original;
    erased.erase(5, 6);  // Remove " World"
    std::cout << "Erased: " << erased << std::endl;  // "Hello"

    // Insert
    std::string inserted = "HelloWorld";
    inserted.insert(5, " ");
    std::cout << "Inserted: " << inserted << std::endl;  // "Hello World"

    std::cout << "\n";

    // =====================
    // Case Conversion
    // =====================
    std::string mixed = "Hello World";

    // To uppercase
    std::string upper = mixed;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    std::cout << "Uppercase: " << upper << std::endl;  // "HELLO WORLD"

    // To lowercase
    std::string lower = mixed;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    std::cout << "Lowercase: " << lower << std::endl;  // "hello world"

    std::cout << "\n";

    // =====================
    // Trimming Whitespace
    // =====================
    std::string padded = "   trim me   ";

    // Trim left
    size_t start = padded.find_first_not_of(" \t\n");
    // Trim right
    size_t end = padded.find_last_not_of(" \t\n");

    std::string trimmed = (start == std::string::npos) ? "" : padded.substr(start, end - start + 1);
    std::cout << "Trimmed: '" << trimmed << "'" << std::endl;  // 'trim me'

    std::cout << "\n";

    // =====================
    // Splitting a String
    // =====================
    std::string csv = "apple,banana,cherry,date";
    std::vector<std::string> tokens;

    // Method 1: Using stringstream with getline
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        tokens.push_back(token);
    }

    std::cout << "Split result:" << std::endl;
    for (const auto& t : tokens) {
        std::cout << "  - " << t << std::endl;
    }

    std::cout << "\n";

    // =====================
    // Joining Strings
    // =====================
    std::vector<std::string> words = {"C++", "is", "awesome"};
    std::string joined;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i > 0) joined += " ";
        joined += words[i];
    }
    std::cout << "Joined: " << joined << std::endl;  // "C++ is awesome"

    std::cout << "\n";

    // =====================
    // Number <-> String Conversion
    // =====================
    // String to number
    int num = std::stoi("42");
    double dbl = std::stod("3.14159");
    std::cout << "Parsed int: " << num << ", double: " << dbl << std::endl;

    // Number to string
    std::string numStr = std::to_string(12345);
    std::string dblStr = std::to_string(2.718);
    std::cout << "Int as string: " << numStr << std::endl;
    std::cout << "Double as string: " << dblStr << std::endl;

    std::cout << "\n";

    // =====================
    // Comparison
    // =====================
    std::string a = "apple";
    std::string b = "banana";

    std::cout << "apple == banana: " << (a == b) << std::endl;  // 0 (false)
    std::cout << "apple < banana: " << (a < b) << std::endl;    // 1 (true, lexicographic)
    std::cout << "compare result: " << a.compare(b) << std::endl;  // negative

    // =====================
    // Reverse
    // =====================
    std::string rev = "desserts";
    std::reverse(rev.begin(), rev.end());
    std::cout << "Reversed: " << rev << std::endl;  // "stressed"

    return 0;
}
