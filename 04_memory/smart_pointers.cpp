#include <iostream>
#include <memory>
#include <string>
#include <vector>

// =====================
// RAII Example Class
// =====================
class Resource {
private:
    std::string name;

public:
    Resource(const std::string& n) : name(n) {
        std::cout << "  [Resource '" << name << "' CREATED]" << std::endl;
    }

    ~Resource() {
        std::cout << "  [Resource '" << name << "' DESTROYED]" << std::endl;
    }

    void use() const {
        std::cout << "  Using resource: " << name << std::endl;
    }

    std::string getName() const { return name; }
};

// =====================
// Functions demonstrating ownership
// =====================

// Transfer ownership INTO function
void takeOwnership(std::unique_ptr<Resource> res) {
    std::cout << "Function owns: " << res->getName() << std::endl;
}  // res destroyed here

// Share ownership
void shareResource(std::shared_ptr<Resource> res) {
    std::cout << "Shared, count: " << res.use_count() << std::endl;
}  // count decremented, not destroyed if others hold it

// Borrow without ownership (raw pointer or reference)
void borrowResource(const Resource& res) {
    res.use();
}  // No ownership, no destruction

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "1. RAW POINTERS (the old, dangerous way)" << std::endl;
    std::cout << "========================================" << std::endl;
    {
        Resource* raw = new Resource("RawPtr");
        raw->use();
        delete raw;  // Must remember to delete! Easy to forget.
        // raw = nullptr;  // Should nullify after delete
    }

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "2. UNIQUE_PTR (single owner, auto-cleanup)" << std::endl;
    std::cout << "========================================" << std::endl;
    {
        // Create with make_unique (preferred, exception-safe)
        std::unique_ptr<Resource> ptr1 = std::make_unique<Resource>("UniqueA");
        ptr1->use();

        // Cannot copy unique_ptr (compile error)
        // std::unique_ptr<Resource> ptr2 = ptr1;  // ERROR!

        // Can MOVE ownership
        std::unique_ptr<Resource> ptr2 = std::move(ptr1);
        std::cout << "After move, ptr1 is: " << (ptr1 ? "valid" : "null") << std::endl;
        ptr2->use();

        // Check before using
        if (ptr2) {
            std::cout << "ptr2 is valid" << std::endl;
        }
    }  // ptr2 auto-deleted here

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "3. SHARED_PTR (multiple owners, ref-counted)" << std::endl;
    std::cout << "========================================" << std::endl;
    {
        std::shared_ptr<Resource> sp1 = std::make_shared<Resource>("SharedB");
        std::cout << "Count after creation: " << sp1.use_count() << std::endl;  // 1

        {
            std::shared_ptr<Resource> sp2 = sp1;  // Copy OK!
            std::cout << "Count after copy: " << sp1.use_count() << std::endl;  // 2

            std::shared_ptr<Resource> sp3 = sp1;
            std::cout << "Count after another copy: " << sp1.use_count() << std::endl;  // 3

            shareResource(sp1);  // Count temporarily increases
        }  // sp2, sp3 go out of scope

        std::cout << "Count after inner scope: " << sp1.use_count() << std::endl;  // 1
    }  // sp1 destroyed, count -> 0, Resource deleted

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "4. WEAK_PTR (observer, no ownership)" << std::endl;
    std::cout << "========================================" << std::endl;
    {
        std::weak_ptr<Resource> weak;

        {
            std::shared_ptr<Resource> shared = std::make_shared<Resource>("WeakC");
            weak = shared;  // weak observes but doesn't own

            std::cout << "Inside scope - expired: " << weak.expired() << std::endl;  // 0 (false)

            // Must lock() to use
            if (auto locked = weak.lock()) {
                locked->use();
                std::cout << "Locked count: " << locked.use_count() << std::endl;  // 2
            }
        }  // shared destroyed

        std::cout << "Outside scope - expired: " << weak.expired() << std::endl;  // 1 (true)

        if (auto locked = weak.lock()) {
            locked->use();  // Won't execute
        } else {
            std::cout << "Resource no longer exists!" << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "5. SMART POINTERS WITH CONTAINERS" << std::endl;
    std::cout << "========================================" << std::endl;
    {
        std::vector<std::unique_ptr<Resource>> resources;

        resources.push_back(std::make_unique<Resource>("Vec1"));
        resources.push_back(std::make_unique<Resource>("Vec2"));
        resources.push_back(std::make_unique<Resource>("Vec3"));

        std::cout << "Resources in vector:" << std::endl;
        for (const auto& r : resources) {
            r->use();
        }

        std::cout << "Clearing vector..." << std::endl;
        resources.clear();
        std::cout << "Vector cleared." << std::endl;
    }

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "SUMMARY: When to Use What" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "unique_ptr: Single owner, default choice" << std::endl;
    std::cout << "            - Function returns dynamically allocated object" << std::endl;
    std::cout << "            - Class owns a resource exclusively" << std::endl;
    std::cout << std::endl;
    std::cout << "shared_ptr: Multiple owners needed" << std::endl;
    std::cout << "            - Shared cache, multiple references to same data" << std::endl;
    std::cout << "            - Observer pattern (with weak_ptr)" << std::endl;
    std::cout << std::endl;
    std::cout << "weak_ptr:   Observe without owning" << std::endl;
    std::cout << "            - Break circular references" << std::endl;
    std::cout << "            - Cache that doesn't prevent cleanup" << std::endl;
    std::cout << std::endl;
    std::cout << "raw ptr:    Non-owning reference (or use reference instead)" << std::endl;
    std::cout << "            - Pointing to stack objects" << std::endl;
    std::cout << "            - Interfacing with C APIs" << std::endl;

    return 0;
}
