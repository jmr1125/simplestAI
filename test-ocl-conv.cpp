#include "convolution.hpp"
#include "matrix.hpp"
#include "ocl.hpp"
#include <iostream>
int main() {
  init();
  matrix a;
  a.setn(5);
  a.setm(5);
  a.m = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 0, 0,
         0,  0,  0,  0,  0,  0, 1, 1, 1, 1, 1, 2};
  /*
    -5 -4 -3 -2 -1
    0 1 2 3 4
    5 0 0 0 0
    0 0 0 0 1
    1 1 1 1 2
   */
  matrix b;
  b.setn(3);
  b.setm(3);
  b.m = {1, 2, 3, 0, 0, 0, 0, 1, 2};
  /*
    1 2 3
    0 0 0
    0 1 2

    2 1 0
    0 0 0
    3 2 1
   */
  std::cout << std::fixed << conv2d(a, b) << std::endl;
  std::cout << std::fixed << conv2d(b, a) << std::endl;
  std::cout << std::fixed << convolution(a, b) << std::endl;
  teardown();
}
