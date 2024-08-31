#include "NN.hpp"
#include "convolution.hpp"
#include "matrix.hpp"
#include "ocl.hpp"
#include <iostream>
#include <ostream>
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

  nnet test;
  test.add_convolution_layer({1, 1}, 5, 5, 3, 3, 0); // 1);
  dynamic_cast<convolution_layer *>(test.last_layer().get())->K[0][0].setn(3);
  dynamic_cast<convolution_layer *>(test.last_layer().get())->K[0][0].setm(3);
  dynamic_cast<convolution_layer *>(test.last_layer().get())->K[0][0].m = {
      1, 2, 0, 0, 1, 0, 3, 2, 1};
  /*
1 2 0
0 1 0
3 2 1

1 2 3
0 1 0
0 2 1
   */
  matrix tmp;
  tmp.setn(5);
  tmp.setm(5);
  tmp.m = {1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 6, 1, 0,
           0, 1, 0, 0, 1, 1, 0, 2, 3, 1, 4, 5};
  std::cout
      << dynamic_cast<convolution_layer *>(test.last_layer().get())->K[0][0]
      << std::endl;
  std::cout << tmp << std::endl;
  std::cout << convolution(
                   dynamic_cast<convolution_layer *>(test.last_layer().get())
                       ->K[0][0],
                   tmp)
            << std::endl;
  test.forward(tmp.m);
  /*
1 2 3 4 5
5 4 3 2 1
6 1 0 0 1
0 0 1 1 0
2 3 1 4 5
  */
  /*
full:
1  4  7  10 13 10 0
5  15 13 11 9  7  0
9  26 20 23 29 17 5
15 28 24 19 12 5  1
20 22 15 8  17 12 1
0  2  6  6  7  6  0
6  13 11 17 24 14 5
pad=1
15 13 .. 7
.  .     .
.     .  .
2  6  .. 6
  */
  std::cout << test.last_layer()->output.size() << std::endl;
  for (const auto x : test.last_layer()->output) {
    std::cout << x << ' ';
  }
  std::cout << std::endl;
  // teardown();
}
