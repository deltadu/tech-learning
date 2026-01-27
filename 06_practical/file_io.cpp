#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main() {
    // =====================
    // Writing to a File
    // =====================
    std::cout << "=== Writing Files ===" << std::endl;

    // ofstream - output file stream
    std::ofstream outFile("example.txt");

    if (outFile.is_open()) {
        outFile << "Hello, File!" << std::endl;
        outFile << "This is line 2." << std::endl;
        outFile << "Numbers: " << 42 << ", " << 3.14 << std::endl;
        outFile.close();
        std::cout << "Wrote to example.txt" << std::endl;
    } else {
        std::cerr << "Failed to open file for writing" << std::endl;
    }

    // Append mode
    std::ofstream appendFile("example.txt", std::ios::app);
    if (appendFile.is_open()) {
        appendFile << "This line was appended." << std::endl;
        appendFile.close();
        std::cout << "Appended to example.txt" << std::endl;
    }

    std::cout << std::endl;

    // =====================
    // Reading from a File
    // =====================
    std::cout << "=== Reading Files ===" << std::endl;

    // ifstream - input file stream
    std::ifstream inFile("example.txt");

    if (inFile.is_open()) {
        std::string line;

        std::cout << "Contents of example.txt:" << std::endl;
        while (std::getline(inFile, line)) {
            std::cout << "  " << line << std::endl;
        }
        inFile.close();
    } else {
        std::cerr << "Failed to open file for reading" << std::endl;
    }

    std::cout << std::endl;

    // =====================
    // Reading Word by Word
    // =====================
    std::cout << "=== Reading Word by Word ===" << std::endl;

    std::ifstream wordFile("example.txt");
    if (wordFile.is_open()) {
        std::string word;
        int count = 0;

        while (wordFile >> word) {
            count++;
        }
        std::cout << "Word count: " << count << std::endl;
        wordFile.close();
    }

    std::cout << std::endl;

    // =====================
    // Reading Entire File at Once
    // =====================
    std::cout << "=== Reading Entire File ===" << std::endl;

    std::ifstream entireFile("example.txt");
    if (entireFile.is_open()) {
        std::stringstream buffer;
        buffer << entireFile.rdbuf();
        std::string contents = buffer.str();

        std::cout << "File length: " << contents.length() << " chars" << std::endl;
        entireFile.close();
    }

    std::cout << std::endl;

    // =====================
    // Reading into Vector
    // =====================
    std::cout << "=== Reading Lines into Vector ===" << std::endl;

    std::ifstream vecFile("example.txt");
    std::vector<std::string> lines;

    if (vecFile.is_open()) {
        std::string line;
        while (std::getline(vecFile, line)) {
            lines.push_back(line);
        }
        vecFile.close();

        std::cout << "Read " << lines.size() << " lines:" << std::endl;
        for (size_t i = 0; i < lines.size(); i++) {
            std::cout << "  [" << i << "] " << lines[i] << std::endl;
        }
    }

    std::cout << std::endl;

    // =====================
    // Writing/Reading Binary Files
    // =====================
    std::cout << "=== Binary Files ===" << std::endl;

    // Write binary
    std::ofstream binOut("data.bin", std::ios::binary);
    if (binOut.is_open()) {
        int numbers[] = {100, 200, 300, 400, 500};
        binOut.write(reinterpret_cast<char*>(numbers), sizeof(numbers));
        binOut.close();
        std::cout << "Wrote binary data" << std::endl;
    }

    // Read binary
    std::ifstream binIn("data.bin", std::ios::binary);
    if (binIn.is_open()) {
        int numbers[5];
        binIn.read(reinterpret_cast<char*>(numbers), sizeof(numbers));
        binIn.close();

        std::cout << "Read binary: ";
        for (int n : numbers) {
            std::cout << n << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;

    // =====================
    // File Position
    // =====================
    std::cout << "=== File Position ===" << std::endl;

    std::ifstream posFile("example.txt");
    if (posFile.is_open()) {
        // Get file size
        posFile.seekg(0, std::ios::end);
        std::streampos fileSize = posFile.tellg();
        std::cout << "File size: " << fileSize << " bytes" << std::endl;

        // Go back to beginning
        posFile.seekg(0, std::ios::beg);

        // Read first 5 characters
        char buffer[6] = {0};
        posFile.read(buffer, 5);
        std::cout << "First 5 chars: " << buffer << std::endl;

        posFile.close();
    }

    std::cout << std::endl;

    // =====================
    // Checking File Existence
    // =====================
    std::cout << "=== File Existence ===" << std::endl;

    std::ifstream checkFile("example.txt");
    if (checkFile.good()) {
        std::cout << "example.txt exists" << std::endl;
        checkFile.close();
    }

    std::ifstream noFile("nonexistent.txt");
    if (!noFile.good()) {
        std::cout << "nonexistent.txt does not exist" << std::endl;
    }

    std::cout << std::endl;

    // =====================
    // CSV Parsing Example
    // =====================
    std::cout << "=== CSV Parsing ===" << std::endl;

    // Create a CSV file
    std::ofstream csvOut("data.csv");
    csvOut << "name,age,city\n";
    csvOut << "Alice,25,NYC\n";
    csvOut << "Bob,30,LA\n";
    csvOut << "Charlie,35,Chicago\n";
    csvOut.close();

    // Parse CSV
    std::ifstream csvIn("data.csv");
    if (csvIn.is_open()) {
        std::string line;

        // Skip header
        std::getline(csvIn, line);

        while (std::getline(csvIn, line)) {
            std::stringstream ss(line);
            std::string name, ageStr, city;

            std::getline(ss, name, ',');
            std::getline(ss, ageStr, ',');
            std::getline(ss, city, ',');

            std::cout << "  Name: " << name << ", Age: " << ageStr << ", City: " << city << std::endl;
        }
        csvIn.close();
    }

    // Cleanup
    std::remove("example.txt");
    std::remove("data.bin");
    std::remove("data.csv");
    std::cout << "\nCleaned up temp files." << std::endl;

    return 0;
}
