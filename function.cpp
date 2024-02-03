#include "function.hpp"
#include "layer.hpp"
#include "main.hpp"
#include "matrix.hpp"
#include <cassert>
#include <cstddef>
#include <iostream>
#ifdef USE_OMP
// #warning omp
#include <omp.h>
#endif
#include <vector>
valT genvalT() {
  static std::random_device rd;
  valT v = rd();
  v -= rd.min();
  v /= (rd.max() - rd.min());
  return v;
}
valT func_x(valT x) { return x; }
valT defunc_x(valT) { return 1; }
valT func_sigma(valT x) { return (valT)1 / (1 + exp(-x)); }
valT defunc_sigma(valT x) { return func_sigma(x) * (1 - func_sigma(x)); }
valT func_tanh(valT x) { return tanh(x); } //(exp(x)-exp(-x))/(exp(x)+exp(-x))
valT defunc_tanh(valT x) { return 1 - func_tanh(x) * func_tanh(x); }
valT func_arctan(valT x) { return atan(x); }
valT defunc_arctan(valT x) { return (valT)1 / (x * x + 1); }
valT func_Softsign(valT x) { return x / (1 + fabs(x)); }
valT defunc_Softsign(valT x) {
  return (valT)1 / ((1 + fabs(x)) * (1 + fabs(x)));
}
valT func_ISRU(valT x, valT alpha) { return x / sqrt(1 + alpha * x * x); }
valT defunc_ISRU(valT x, valT alpha) {
  return (valT)1 / pow(sqrt(1 + alpha * x * x), 3);
}
valT func_ReLU(valT x) { return x < 0 ? 0 : x; }
valT defunc_ReLU(valT x) { return x < 0 ? 0 : 1; }
valT func_Leaky_ReLU(valT x) { return x < 0 ? 0.01 * x : x; }
valT defunc_Leaky_ReLU(valT x) { return x < 0 ? 0.01 : 1; }
valT func_PReLU(valT x, valT alpha) { return x < 0 ? alpha * x : x; }
valT defunc_PReLU(valT x, valT alpha) { return x < 0 ? alpha : 1; }
// RReLU?
valT func_ELU(valT x, valT alpha) { return x < 0 ? alpha * (exp(x) - 1) : x; }
valT defunc_ELU(valT x, valT alpha) {
  return x < 0 ? alpha + func_ELU(x, alpha) : 1;
}
valT func_SELU(valT x) { return 1.0507 * func_ELU(x, 1.67326); }
valT defunc_SELU(valT x) { return 1.0507 * (x < 0 ? 1.67326 * exp(x) : 1); }
valT func_softmax(VvalT x, int i) {
  valT a = exp(x[i]);
  valT b = 0;
  for (int i = 0; i < x.size(); ++i) {
    b += exp(x[i]);
  }
  return a / b;
}
