#include "main.hpp"
struct adam {
  adam(int size=0);
  VvalT update(VvalT g, valT alpha, int total);
  VvalT m, v, vh, mh;
  int size;
  static constexpr double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
};
