#include "average_layer.hpp"
#include "main.hpp"
#include "matrix.hpp"
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

average_layer::~average_layer() {}

void average_layer::init(std::random_device &&) {}
void average_layer::set_IOsize(int isize, int osize) {
  if (i_n * i_m != isize || (i_n / 2) * (i_m / 2) != osize) {
    throw std::runtime_error(
        "init average_layer : io: " + std::to_string(isize) + " , " +
        std::to_string(osize) + " nm: " + std::to_string(i_n) + " , " +
        std::to_string(i_m));
  }
  output.resize(i_n / 2 * i_m / 2);
  Isize = isize;
  Osize = osize;
}
VvalT average_layer::forward(const vector<valT> &in) {
  for (int i = 0; i < i_n / 2; ++i) {
    for (int j = 0; j < i_m / 2; ++j) {
      output[i * i_m / 2 + j] =
          (in[i * i_m + j] + in[i * i_m + j + 1] + in[(i + 1) * i_m + j] +
           in[(i + 1) * i_m + j + 1]) /
          4;
    }
  }
  return output;
}
VvalT average_layer::backward(const VvalT &grad) const {
  matrix res;
  res.setn(i_n);
  res.setm(i_m);
  for (int i = 0; i < i_n; ++i) {
    for (int j = 0; j < i_m; ++j) {
      res(i, j) = grad[(i / 2) * i_m / 2 + j / 2] / 4;
    }
  }
  return res.m;
}
VvalT average_layer::update(const VvalT &, const VvalT &, double lr) const {
  return {};
}
void average_layer::update(VvalT::const_iterator &) { return; }

void average_layer::save(ostream &o) const {
  o << i_n << " " << i_m << std::endl;
}
void average_layer::load(std::istream &i) {
  i >> i_n >> i_m;
  set_IOsize(i_n * i_m, i_n / 2 * i_m / 2);
}
