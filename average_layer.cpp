#include "average_layer.hpp"
#include "main.hpp"
#include "matrix.hpp"
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

average_layer::~average_layer() {}

void average_layer::init(std::random_device &&) {}
void average_layer::set_IOsize(int isize, int osize) {
  if (i_n * i_m * Ichannels != isize ||
      ceil(1.0 * i_n / size) * ceil(1.0 * i_m / size) * Ochannels != osize) {
    throw std::runtime_error(
        "init average_layer : io: " + std::to_string(isize) + " , " +
        std::to_string(osize) + " nm: " + std::to_string(i_n) + " , " +
        std::to_string(i_m));
  }
  // if (i_n % size || i_m % size) {
  //   throw std::runtime_error("init average_layer : nm: " + std::to_string(i_n) +
  //                            " " + std::to_string(i_m));
  // }
  output.resize(ceil(1.0 * i_n / size) * ceil(1.0 * i_m / size) * Ichannels);
  Isize = isize;
  Osize = osize;
}
VvalT average_layer::forward(const vector<valT> &in) {
  const int o_n = ceil(1.0 * i_n / size);
  const int o_m = ceil(1.0 * i_m / size);
  // for (int c = 0; c < Ichannels; ++c)
  //   for (int i = 0; i < o_n; ++i) {
  //     for (int j = 0; j < o_m; ++j) {
  //       output[i * o_m + j + c * o_m * o_n] = 0;
  //       for (int dx = 0; dx < size; ++dx)
  //         for (int dy = 0; dy < size; ++dy) {
  //           output.at(i * o_m + j + c * o_m * o_n) +=
  //               (i * size + dx >= i_n || j * size + dy >= i_m)
  //                   ? 0
  //                   : in.at((i * size + dx) * i_n + j * size + dy +
  //                           c * i_n * i_m) /
  //                         size / size;
  //         }
  //     }
  //   }
  fill(output.begin(), output.end(), 0);
  for (int c = 0; c < Ichannels; ++c) {
    for (int i = 0; i < i_n; ++i) {
      for (int j = 0; j < i_m; ++j) {
        output.at((i / size) * o_m + j / size + c * o_n * o_m) +=
            in.at(i * i_m + j + c * i_n * i_m) / size / size;
      }
    }
  }
  return output;
}
VvalT average_layer::backward(const VvalT &grad) const {
  const int o_n = ceil(1.0 * i_n / size);
  const int o_m = ceil(1.0 * i_m / size);
  VvalT res;
  res.resize(i_m * i_n * Ichannels);
  for (int c = 0; c < Ichannels; ++c)
    for (int i = 0; i < i_n; ++i) {
      for (int j = 0; j < i_m; ++j) {
        res.at(j + i * i_m + c * i_n * i_m) =
            grad.at((i / size) * o_m + j / size + c * o_n * o_m) / size / size;
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
void average_layer::randomize_nan(std::random_device &&) { return; }
