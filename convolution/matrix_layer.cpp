#include "matrix_layer.hpp"
#include <random>

matrix_layer::~matrix_layer() {}
void matrix_layer::init(std::random_device &&rd) {
  for (valT &x : M.m) {
    x = ((rd() + rd.min()) * 1.0 / rd.max() * 2 - 1) * sqrt(Isize);
  }
}
void matrix_layer::set_IOsize(int isize, int osize) {
  M.setn(osize);
  M.setm(isize);
  Isize = isize;
  Osize = osize;
}
vector<valT> matrix_layer::forward(const vector<valT> &input) {
  output = M * input;
  return output;
}
vector<valT> matrix_layer::backward(const vector<valT> &grad) {
  return M.T() * grad;
}
void matrix_layer::update(const vector<valT> &grad, const vector<valT> &input,
                          double lr) {
  for (int i = 0; i < M.getn(); i++)
    for (int j = 0; j < M.getm(); j++)
      M(i, j) -= lr * grad[i] * input[j];
}
