#include "../../src/SolverApp.h"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        return SolverApp{}.run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
