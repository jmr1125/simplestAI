#include "max_layer.hpp"
#include "main.hpp"
#include "matrix.hpp"
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

max_layer::~max_layer() {}

void max_layer::init(std::random_device &&) {}
void max_layer::set_IOsize(int isize, int osize) {
  if (i_n * i_m * Ichannels != isize ||
      (i_n / 2) * (i_m / 2) * Ochannels != osize) {
    throw std::runtime_error("init max_layer : io: " + std::to_string(isize) +
                             " , " + std::to_string(osize) + " nm: " +
                             std::to_string(i_n) + " , " + std::to_string(i_m));
  }
  output.resize(i_n / 2 * i_m / 2 * Ichannels);
  Isize = isize;
  Osize = osize;
}
VvalT max_layer::forward(const vector<valT> &in) {
  const int o_n = i_n / 2;
  const int o_m = i_m / 2;
  input = in;
  for (int c = 0; c < Ichannels; ++c)
    for (int i = 0; i < i_n / 2; ++i) {
      for (int j = 0; j < i_m / 2; ++j) {
        output[i * o_m + j + c * o_m * o_n] =
            std::max(std::max(in[i * i_m + j + c * i_m * i_n],
                              in[i * i_m + j + 1 + j + c * i_m * i_n]),
                     std::max(in[(i + 1) * i_m + j + j + c * i_m * i_n],
                              in[(i + 1) * i_m + j + 1 + j + c * i_m * i_n]));
      }
    }
  return output;
}
VvalT max_layer::backward(const VvalT &grad) const {
  const int o_n = i_n / 2;
  const int o_m = i_m / 2;
  VvalT res;
  res.resize(i_n * i_m * Ichannels);
  for (int c = 0; c < Ichannels; ++c)
    for (int i = 0; i < i_n; ++i) {
      for (int j = 0; j < i_m; ++j) {
        if (input[i * i_m + j] == output[(i / 2) * o_m + j / 2])
          res[i * i_m + j + c * i_m * i_n] =
              grad[(i / 2) * o_m + j / 2 + c * o_m * o_n];
        else
          res[i * i_m + j + c * i_m * i_n] = 0;
      }
    }
  return res;
}
VvalT max_layer::update(const VvalT &, const VvalT &, double lr) const {
  return {};
}
void max_layer::update(VvalT::const_iterator &) { return; }

void max_layer::save(ostream &o) const {
  o << i_n << " " << i_m << " " << Ichannels << std::endl;
}
void max_layer::load(std::istream &i) {
  i >> i_n >> i_m >> Ichannels;
  Ochannels = Ichannels;
  set_IOsize(i_n * i_m * Ichannels, i_n / 2 * i_m / 2 * Ichannels);
}
