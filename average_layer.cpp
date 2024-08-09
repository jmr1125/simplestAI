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
  if (i_n * i_m != isize ||
      ceil(1.0 * i_n / size) * ceil(1.0 * i_m / size) * Ochannels != osize) {
    throw std::runtime_error(
        "init average_layer : io: " + std::to_string(isize) + " , " +
        std::to_string(osize) + " nm: " + std::to_string(i_n) + " , " +
        std::to_string(i_m));
  }
  output.resize(ceil(1.0 * i_n / size) * ceil(1.0 * i_m / size) * Ichannels);
  Isize = isize;
  Osize = osize;
}
VvalT average_layer::forward(const vector<valT> &in) {
  const int o_n = i_n / 2;
  const int o_m = i_m / 2;
  for (int c = 0; c < Ichannels; ++c)
    for (int i = 0; i < i_n / 2; ++i) {
      for (int j = 0; j < i_m / 2; ++j) {
        output[i * o_m + j + c * o_m * o_n] = 0;
        for (int dx = 0; dx < size; ++dx)
          for (int dy = 0; dy < size; ++dy) {
            output[i * o_m + j + c * o_m * o_n] +=
                (i * size + dx >= i_n || j * size + dy >= i_m)
                    ? 0
                    : in[(i * size + dx) * i_n + j * size + dy + c * i_n * i_m];
          }
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
            grad[(i / size) * o_m + j / size + c * o_n * o_m] / size / size;
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

std::shared_ptr<layer> average_layer::clone() const {
  return std::make_shared<average_layer>(*this);
}
