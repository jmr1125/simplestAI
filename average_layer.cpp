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
  output.resize(i_n / 2 * i_m / 2 * Ichannels);
  Isize = isize;
  Osize = osize;
}
VvalT average_layer::forward(const vector<valT> &in) {
  const int o_n = i_n / 2;
  const int o_m = i_m / 2;
  for (int c = 0; c < Ichannels; ++c)
    for (int i = 0; i < i_n / 2; ++i) {
      for (int j = 0; j < i_m / 2; ++j) {
        output[i * o_m + j + c * o_n * o_m] =
            (in[i * i_m + j + c * i_n * i_m] +
             in[i * i_m + (j + 1) + c * i_n * i_m] +
             in[(i + 1) * i_m + j + c * i_n * i_m] +
             in[(i + 1) * i_m + (j + 1) + c * i_n * i_m]) /
            4;
      }
    }
  return output;
}
VvalT average_layer::backward(const VvalT &grad) const {
  const int o_n = i_n / 2;
  const int o_m = i_m / 2;
  VvalT res;
  res.resize(i_m * i_n * Ichannels);
  for (int c = 0; c < Ichannels; ++c)
    for (int i = 0; i < i_n; ++i) {
      for (int j = 0; j < i_m; ++j) {
        res[j + i * i_m + c * i_n * i_m] =
            grad[(i / 2) * o_m + j / 2 + c * o_n * o_m] / 4;
      }
    }
  return res;
}
VvalT average_layer::update(const VvalT &, const VvalT &) const { return {}; }
void average_layer::update(VvalT::const_iterator &) { return; }

void average_layer::save(ostream &o) const {
  o << i_n << " " << i_m << " " << Ichannels << std::endl;
}
void average_layer::load(std::istream &i) {
  i >> i_n >> i_m >> Ichannels;
  Ochannels = Ichannels;
  set_IOsize(i_n * i_m * Ichannels, i_n / 2 * i_m / 2 * Ichannels);
}

size_t average_layer::get_varnum() const { return 0; }
