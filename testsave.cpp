#include "network.h"
#include <cstddef>
#include <fstream>
#include <iostream>
int main() {
  network n1({1, 3, 3, 4}, NULL, NULL);
  {
    std::ofstream ofp("testsave.net");
    n1.save(ofp);
  }
  {
    std::ifstream ifp("testsave.net");
    network n2({}, NULL, NULL);
    n1.load(ifp);
    n1.save(std::cout);
  }
}
