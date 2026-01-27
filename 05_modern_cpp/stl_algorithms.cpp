#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>    // accumulate, iota
#include <string>

void printVec(const std::string& label, const std::vector<int>& v) {
    std::cout << label << ": ";
    for (int n : v) std::cout << n << " ";
    std::cout << std::endl;
}

int main() {
    // =====================
    // Sorting
    // =====================
    std::cout << "=== Sorting ===" << std::endl;

    std::vector<int> nums = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    printVec("Original", nums);

    std::sort(nums.begin(), nums.end());
    printVec("Sorted (asc)", nums);

    std::sort(nums.begin(), nums.end(), std::greater<int>());
    printVec("Sorted (desc)", nums);

    // Partial sort - only first 3 elements sorted
    std::vector<int> partial = {5, 2, 8, 1, 9, 3};
    std::partial_sort(partial.begin(), partial.begin() + 3, partial.end());
    printVec("Partial sort (3)", partial);

    std::cout << std::endl;

    // =====================
    // Searching
    // =====================
    std::cout << "=== Searching ===" << std::endl;

    std::vector<int> sorted = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    // find - linear search, returns iterator
    auto it = std::find(sorted.begin(), sorted.end(), 5);
    if (it != sorted.end()) {
        std::cout << "Found 5 at index: " << (it - sorted.begin()) << std::endl;
    }

    // binary_search - requires sorted container, returns bool
    bool found = std::binary_search(sorted.begin(), sorted.end(), 7);
    std::cout << "Binary search for 7: " << (found ? "found" : "not found") << std::endl;

    // lower_bound - first element >= value
    auto lb = std::lower_bound(sorted.begin(), sorted.end(), 5);
    std::cout << "Lower bound of 5: " << *lb << " at index " << (lb - sorted.begin()) << std::endl;

    // find_if - find with predicate
    auto even = std::find_if(sorted.begin(), sorted.end(), [](int x) { return x % 2 == 0; });
    std::cout << "First even: " << *even << std::endl;

    std::cout << std::endl;

    // =====================
    // Counting
    // =====================
    std::cout << "=== Counting ===" << std::endl;

    std::vector<int> data = {1, 2, 2, 3, 2, 4, 2, 5};

    int count2 = std::count(data.begin(), data.end(), 2);
    std::cout << "Count of 2s: " << count2 << std::endl;

    int countEven = std::count_if(data.begin(), data.end(), [](int x) { return x % 2 == 0; });
    std::cout << "Count of evens: " << countEven << std::endl;

    std::cout << std::endl;

    // =====================
    // Transforming
    // =====================
    std::cout << "=== Transform ===" << std::endl;

    std::vector<int> src = {1, 2, 3, 4, 5};
    std::vector<int> dst(src.size());

    // Double each element
    std::transform(src.begin(), src.end(), dst.begin(), [](int x) { return x * 2; });
    printVec("Original", src);
    printVec("Doubled", dst);

    // In-place transform
    std::transform(src.begin(), src.end(), src.begin(), [](int x) { return x * x; });
    printVec("Squared (in-place)", src);

    std::cout << std::endl;

    // =====================
    // Accumulate / Reduce
    // =====================
    std::cout << "=== Accumulate ===" << std::endl;

    std::vector<int> values = {1, 2, 3, 4, 5};

    int sum = std::accumulate(values.begin(), values.end(), 0);
    std::cout << "Sum: " << sum << std::endl;  // 15

    int product = std::accumulate(values.begin(), values.end(), 1, std::multiplies<int>());
    std::cout << "Product: " << product << std::endl;  // 120

    // Custom accumulator - concatenate strings
    std::vector<std::string> words = {"Hello", " ", "World", "!"};
    std::string sentence = std::accumulate(words.begin(), words.end(), std::string(""));
    std::cout << "Sentence: " << sentence << std::endl;

    std::cout << std::endl;

    // =====================
    // Min / Max
    // =====================
    std::cout << "=== Min / Max ===" << std::endl;

    std::vector<int> minmax = {3, 1, 4, 1, 5, 9, 2, 6};

    auto minIt = std::min_element(minmax.begin(), minmax.end());
    auto maxIt = std::max_element(minmax.begin(), minmax.end());
    std::cout << "Min: " << *minIt << " at index " << (minIt - minmax.begin()) << std::endl;
    std::cout << "Max: " << *maxIt << " at index " << (maxIt - minmax.begin()) << std::endl;

    auto [minE, maxE] = std::minmax_element(minmax.begin(), minmax.end());
    std::cout << "Minmax: " << *minE << ", " << *maxE << std::endl;

    std::cout << std::endl;

    // =====================
    // Removing / Filtering
    // =====================
    std::cout << "=== Remove / Erase ===" << std::endl;

    std::vector<int> rem = {1, 2, 3, 2, 4, 2, 5};
    printVec("Before remove", rem);

    // remove doesn't resize - use erase-remove idiom
    rem.erase(std::remove(rem.begin(), rem.end(), 2), rem.end());
    printVec("After removing 2s", rem);

    // remove_if with predicate
    std::vector<int> rem2 = {1, 2, 3, 4, 5, 6, 7, 8};
    rem2.erase(std::remove_if(rem2.begin(), rem2.end(), [](int x) { return x % 2 == 0; }), rem2.end());
    printVec("After removing evens", rem2);

    std::cout << std::endl;

    // =====================
    // Copying / Filling
    // =====================
    std::cout << "=== Copy / Fill ===" << std::endl;

    std::vector<int> source = {1, 2, 3, 4, 5};
    std::vector<int> dest(5);

    std::copy(source.begin(), source.end(), dest.begin());
    printVec("Copied", dest);

    std::fill(dest.begin(), dest.end(), 42);
    printVec("Filled with 42", dest);

    // iota - fill with incrementing values
    std::iota(dest.begin(), dest.end(), 10);
    printVec("iota from 10", dest);

    std::cout << std::endl;

    // =====================
    // Reversing / Rotating
    // =====================
    std::cout << "=== Reverse / Rotate ===" << std::endl;

    std::vector<int> rev = {1, 2, 3, 4, 5};
    std::reverse(rev.begin(), rev.end());
    printVec("Reversed", rev);

    std::vector<int> rot = {1, 2, 3, 4, 5};
    std::rotate(rot.begin(), rot.begin() + 2, rot.end());  // Rotate left by 2
    printVec("Rotated left 2", rot);

    std::cout << std::endl;

    // =====================
    // All/Any/None
    // =====================
    std::cout << "=== All / Any / None ===" << std::endl;

    std::vector<int> check = {2, 4, 6, 8, 10};

    bool allEven = std::all_of(check.begin(), check.end(), [](int x) { return x % 2 == 0; });
    bool anyOdd = std::any_of(check.begin(), check.end(), [](int x) { return x % 2 != 0; });
    bool noneNeg = std::none_of(check.begin(), check.end(), [](int x) { return x < 0; });

    std::cout << "All even: " << allEven << std::endl;   // 1
    std::cout << "Any odd: " << anyOdd << std::endl;     // 0
    std::cout << "None negative: " << noneNeg << std::endl;  // 1

    return 0;
}
