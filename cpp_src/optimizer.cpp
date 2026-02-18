#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <map>

namespace py = pybind11;

class BlockManager {
public:
    BlockManager(int num_blocks, int block_size) 
        : free_blocks(num_blocks), block_size(block_size) {
        for (int i = 0; i < num_blocks; ++i) {
            free_list.push_back(i);
        }
    }

    // Maps logical token indices to physical blocks
    std::vector<int> allocate(int num_tokens) {
        int blocks_needed = (num_tokens + block_size - 1) / block_size;
        std::vector<int> allocated;
        for (int i = 0; i < blocks_needed && !free_list.empty(); ++i) {
            allocated.push_back(free_list.back());
            free_list.pop_back();
        }
        return allocated;
    }

private:
    int free_blocks;
    int block_size;
    std::vector<int> free_list;
};

PYBIND11_MODULE(nexus_cpp, m) {
    py::class_<BlockManager>(m, "BlockManager")
        .def(py::init<int, int>())
        .def("allocate", &BlockManager::allocate);
}
